from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
from .mock_runtime import FAILURE_MODE_BY_QID, actor_answer, evaluator, reflector
from .schemas import AttemptTrace, QAExample, ReflectionEntry, RunRecord

import os
import time

@dataclass
class BaseAgent:
    agent_type: Literal["react", "reflexion"]
    max_attempts: int = 1
    def run(self, example: QAExample) -> RunRecord:
        reflection_memory: list[str] = []
        reflections: list[ReflectionEntry] = []
        traces: list[AttemptTrace] = []
        final_answer = ""
        final_score = 0
        for attempt_id in range(1, self.max_attempts + 1):
            from .mock_runtime import LAST_ACTOR_METRICS
            LAST_ACTOR_METRICS["token_estimate"] = 0
            LAST_ACTOR_METRICS["latency_ms"] = 0
            
            start_time = time.perf_counter()
            answer = actor_answer(example, attempt_id, self.agent_type, reflection_memory)
            actor_latency = int((time.perf_counter() - start_time) * 1000)
            
            evaluator_start = time.perf_counter()
            judge = evaluator(example, answer)
            evaluator_latency = int((time.perf_counter() - evaluator_start) * 1000)
            
            if os.getenv("LLM_MODE", "mock") == "mock":
                token_estimate = 320 + (attempt_id * 65) + (120 if self.agent_type == "reflexion" else 0)
                latency_ms = 160 + (attempt_id * 40) + (90 if self.agent_type == "reflexion" else 0)
            else:
                token_estimate = LAST_ACTOR_METRICS.get("token_estimate", 0) + getattr(judge, "token_estimate", 0)
                latency_ms = LAST_ACTOR_METRICS.get("latency_ms", 0) + getattr(judge, "latency_ms", 0)
                
            trace = AttemptTrace(
                attempt_id=attempt_id, 
                answer=answer, 
                score=judge.score, 
                reason=judge.reason, 
                token_estimate=token_estimate, 
                latency_ms=latency_ms
            )
            final_answer = answer
            final_score = judge.score
            
            if judge.score == 1:
                traces.append(trace)
                break
            
            if self.agent_type == "reflexion" and attempt_id < self.max_attempts:
                # Call reflector
                reflector_start = time.perf_counter()
                reflection = reflector(example, attempt_id, judge)
                reflector_latency = int((time.perf_counter() - reflector_start) * 1000)
                
                reflections.append(reflection)
                trace.reflection = reflection
                
                if os.getenv("LLM_MODE", "mock") == "mock":
                    trace.token_estimate += 150
                    trace.latency_ms += 110
                else:
                    trace.token_estimate += getattr(reflection, "token_estimate", 0)
                    trace.latency_ms += getattr(reflection, "latency_ms", 0)
                    
                # Update reflection memory
                reflection_memory.append(
                    f"Attempt {attempt_id} failed. Reason: {judge.reason}. Lesson: {reflection.lesson}. Next strategy: {reflection.next_strategy}"
                )
            
            traces.append(trace)
        total_tokens = sum(t.token_estimate for t in traces)
        total_latency = sum(t.latency_ms for t in traces)
        failure_mode = "none" if final_score == 1 else FAILURE_MODE_BY_QID.get(example.qid, "wrong_final_answer")
        return RunRecord(qid=example.qid, question=example.question, gold_answer=example.gold_answer, agent_type=self.agent_type, predicted_answer=final_answer, is_correct=bool(final_score), attempts=len(traces), token_estimate=total_tokens, latency_ms=total_latency, failure_mode=failure_mode, reflections=reflections, traces=traces)

class ReActAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(agent_type="react", max_attempts=1)

class ReflexionAgent(BaseAgent):
    def __init__(self, max_attempts: int = 3) -> None:
        super().__init__(agent_type="reflexion", max_attempts=max_attempts)
