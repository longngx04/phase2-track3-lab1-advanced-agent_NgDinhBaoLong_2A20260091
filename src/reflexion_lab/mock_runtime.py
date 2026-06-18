from __future__ import annotations
import os
import time
import json
from google import genai
from google.genai import types
from openai import OpenAI

from .schemas import QAExample, JudgeResult, ReflectionEntry
from .utils import normalize_answer

FIRST_ATTEMPT_WRONG = {"hp2": "London", "hp4": "Atlantic Ocean", "hp6": "Red Sea", "hp8": "Andes"}
FAILURE_MODE_BY_QID = {"hp2": "incomplete_multi_hop", "hp4": "wrong_final_answer", "hp6": "entity_drift", "hp8": "entity_drift"}

LAST_ACTOR_METRICS = {"token_estimate": 0, "latency_ms": 0}

def get_mock_wrong_answer(example: QAExample) -> str | None:
    if example.qid in FIRST_ATTEMPT_WRONG:
        return FIRST_ATTEMPT_WRONG[example.qid]
    
    try:
        h = sum(ord(c) for c in example.qid)
    except Exception:
        h = 0
        
    # Every 5th question will fail first attempt to generate diverse failures
    if h % 5 == 0:
        if h % 3 == 0:
            FAILURE_MODE_BY_QID[example.qid] = "incomplete_multi_hop"
            return "London"
        elif h % 3 == 1:
            FAILURE_MODE_BY_QID[example.qid] = "wrong_final_answer"
            return "Atlantic Ocean"
        else:
            FAILURE_MODE_BY_QID[example.qid] = "entity_drift"
            return "Andes"
    return None

def call_llm(system_prompt: str, user_prompt: str, json_mode: bool = False) -> tuple[str, int, int]:
    provider = os.getenv("LLM_PROVIDER", "gemini").lower()
    
    max_retries = 5
    backoff_factor = 2.0
    initial_delay = 2.0
    
    for attempt in range(max_retries):
        start_time = time.perf_counter()
        try:
            if provider == "gemini":
                api_key = os.getenv("GEMINI_API_KEY")
                if not api_key:
                    raise ValueError("GEMINI_API_KEY environment variable is not set.")
                    
                client = genai.Client(api_key=api_key)
                model_name = os.getenv("LLM_MODEL", "gemini-2.5-flash")
                
                config = types.GenerateContentConfig()
                if json_mode:
                    config.response_mime_type = "application/json"
                    
                response = client.models.generate_content(
                    model=model_name,
                    contents=[
                        types.Content(
                            role="user",
                            parts=[types.Part.from_text(text=f"{system_prompt}\n\nUser Question/Input:\n{user_prompt}")]
                        )
                    ],
                    config=config
                )
                latency = int((time.perf_counter() - start_time) * 1000)
                
                token_count = 0
                if response.usage_metadata:
                    token_count = response.usage_metadata.total_token_count
                if token_count == 0:
                    token_count = (len(system_prompt) + len(user_prompt) + len(response.text)) // 4
                    
                # Introduce a small sleep to avoid rate limits on the next call
                time.sleep(1.0)
                return response.text, token_count, latency
                
            else:
                api_key = os.getenv("OPENAI_API_KEY")
                base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
                model_name = os.getenv("LLM_MODEL", "gpt-4o-mini")
                
                if not api_key:
                    raise ValueError("OPENAI_API_KEY environment variable is not set.")
                    
                client = OpenAI(api_key=api_key, base_url=base_url)
                
                response_format = None
                if json_mode:
                    response_format = {"type": "json_object"}
                    
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    response_format=response_format
                )
                latency = int((time.perf_counter() - start_time) * 1000)
                token_count = response.usage.total_tokens if response.usage else (len(system_prompt) + len(user_prompt) + len(response.choices[0].message.content)) // 4
                
                time.sleep(1.0)
                return response.choices[0].message.content, token_count, latency
                
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
                
            err_msg = str(e).lower()
            # If it is a bad credentials, bad request, or unauthorized error, do not retry
            if any(x in err_msg for x in ["api_key", "unauthorized", "invalid api key", "forbidden", "permission", "400 bad request", "403 forbidden", "404 not found"]):
                raise e
                
            delay = initial_delay * (backoff_factor ** attempt)
            print(f"[Warning] LLM call failed: {e}. Retrying in {delay:.1f} seconds...")
            time.sleep(delay)

def actor_answer(example: QAExample, attempt_id: int, agent_type: str, reflection_memory: list[str]) -> str:
    if os.getenv("LLM_MODE", "mock") == "mock":
        wrong_ans = get_mock_wrong_answer(example)
        if wrong_ans is None:
            return example.gold_answer
        if agent_type == "react":
            return wrong_ans
        if attempt_id == 1 and not reflection_memory:
            return wrong_ans
        return example.gold_answer
    else:
        from .prompts import ACTOR_SYSTEM
        context_str = "\n\n".join([f"Document: {c.title}\n{c.text}" for c in example.context])
        reflection_str = ""
        if reflection_memory:
            reflection_str = "\nReflection History of past failures:\n" + "\n".join(f"- {r}" for r in reflection_memory)
            
        user_prompt = f"Context Documents:\n{context_str}\n\nQuestion: {example.question}{reflection_str}"
        
        response_text, tokens, latency = call_llm(ACTOR_SYSTEM, user_prompt, json_mode=False)
        
        LAST_ACTOR_METRICS["token_estimate"] = tokens
        LAST_ACTOR_METRICS["latency_ms"] = latency
        
        ans = response_text
        if "Final Answer:" in response_text:
            ans = response_text.split("Final Answer:")[-1].strip()
        elif "final answer:" in response_text.lower():
            ans = response_text.lower().split("final answer:")[-1].strip()
        return ans

def evaluator(example: QAExample, answer: str) -> JudgeResult:
    if os.getenv("LLM_MODE", "mock") == "mock":
        if normalize_answer(example.gold_answer) == normalize_answer(answer):
            return JudgeResult(score=1, reason="Final answer matches the gold answer after normalization.")
        
        # Check dynamic failure first
        wrong_ans = get_mock_wrong_answer(example)
        if wrong_ans is not None:
            try:
                h = sum(ord(c) for c in example.qid)
            except Exception:
                h = 0
            if h % 3 == 0:
                return JudgeResult(score=0, reason="The answer stopped at the birthplace city and never completed the second hop to the river.", missing_evidence=["Need to identify the river that flows through London."], spurious_claims=[])
            elif h % 3 == 1:
                return JudgeResult(score=0, reason="The final answer selected the wrong second-hop entity.", missing_evidence=["Need to ground the answer in the second paragraph."], spurious_claims=[answer])
            else:
                return JudgeResult(score=0, reason="The final answer selected the wrong second-hop entity.", missing_evidence=["Need to ground the answer in the second paragraph."], spurious_claims=[answer])
        
        # Fallback for original hp questions
        if normalize_answer(answer) == "london":
            return JudgeResult(score=0, reason="The answer stopped at the birthplace city and never completed the second hop to the river.", missing_evidence=["Need to identify the river that flows through London."], spurious_claims=[])
        return JudgeResult(score=0, reason="The final answer selected the wrong second-hop entity.", missing_evidence=["Need to ground the answer in the second paragraph."], spurious_claims=[answer])
    else:
        from .prompts import EVALUATOR_SYSTEM
        user_prompt = f"Question: {example.question}\nGold Answer: {example.gold_answer}\nPredicted Answer: {answer}"
        
        response_text, tokens, latency = call_llm(EVALUATOR_SYSTEM, user_prompt, json_mode=True)
        
        try:
            clean_text = response_text.strip()
            if clean_text.startswith("```json"):
                clean_text = clean_text.split("```json")[1].split("```")[0].strip()
            elif clean_text.startswith("```"):
                clean_text = clean_text.split("```")[1].split("```")[0].strip()
                
            data = json.loads(clean_text)
            judge = JudgeResult.model_validate(data)
            judge.token_estimate = tokens
            judge.latency_ms = latency
            return judge
        except Exception as e:
            score = 1 if normalize_answer(example.gold_answer) == normalize_answer(answer) else 0
            return JudgeResult(
                score=score, 
                reason=f"Failed to parse evaluator response: {response_text}. Error: {str(e)}", 
                token_estimate=tokens, 
                latency_ms=latency
            )

def reflector(example: QAExample, attempt_id: int, judge: JudgeResult) -> ReflectionEntry:
    if os.getenv("LLM_MODE", "mock") == "mock":
        try:
            val = sum(ord(c) for c in example.qid)
        except Exception:
            val = 0
        strategy = "Do the second hop explicitly: birthplace city -> river through that city." if val % 7 == 0 else "Verify the final entity against the second paragraph before answering."
        return ReflectionEntry(attempt_id=attempt_id, failure_reason=judge.reason, lesson="A partial first-hop answer is not enough; the final answer must complete all hops.", next_strategy=strategy)
    else:
        from .prompts import REFLECTOR_SYSTEM
        user_prompt = f"Question: {example.question}\nAttempt ID: {attempt_id}\nEvaluator Feedback: {judge.reason}"
        
        response_text, tokens, latency = call_llm(REFLECTOR_SYSTEM, user_prompt, json_mode=True)
        
        try:
            clean_text = response_text.strip()
            if clean_text.startswith("```json"):
                clean_text = clean_text.split("```json")[1].split("```")[0].strip()
            elif clean_text.startswith("```"):
                clean_text = clean_text.split("```")[1].split("```")[0].strip()
                
            data = json.loads(clean_text)
            reflection = ReflectionEntry.model_validate(data)
            reflection.token_estimate = tokens
            reflection.latency_ms = latency
            return reflection
        except Exception as e:
            return ReflectionEntry(
                attempt_id=attempt_id,
                failure_reason=judge.reason,
                lesson=f"Failed to parse reflector response: {response_text}. Error: {str(e)}",
                next_strategy="Carefully re-read the context documents and double check the entity hops.",
                token_estimate=tokens,
                latency_ms=latency
            )
