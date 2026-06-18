import os
import zipfile
import json

def main():
    zip_path = "data/hotpot_100_temp.json"
    dest_path = "data/hotpot_100.json"
    
    if not os.path.exists(zip_path):
        print(f"Error: {zip_path} does not exist.")
        return
        
    print("Extracting and parsing dataset...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        with z.open('hotpot_dev_distractor_v1.json') as f:
            data = json.load(f)
            
    print(f"Loaded {len(data)} items from zip.")
    
    formatted = []
    # Take the first 100 examples
    for item in data[:100]:
        qid = item["_id"]
        difficulty = item.get("level", "medium")
        if difficulty not in ["easy", "medium", "hard"]:
            difficulty = "medium"
        question = item["question"]
        gold_answer = item["answer"]
        
        context_chunks = []
        for title, sentences in item["context"]:
            text = "".join(sentences)
            context_chunks.append({
                "title": title,
                "text": text
            })
            
        formatted.append({
            "qid": qid,
            "difficulty": difficulty,
            "question": question,
            "gold_answer": gold_answer,
            "context": context_chunks
        })
        
    with open(dest_path, "w", encoding="utf-8") as f:
        json.dump(formatted, f, indent=2, ensure_ascii=False)
        
    print(f"Saved {len(formatted)} formatted examples to {dest_path}.")
    
    try:
        os.remove(zip_path)
        print(f"Cleaned up temporary file {zip_path}.")
    except Exception as e:
        print(f"Could not remove {zip_path}: {e}")

if __name__ == "__main__":
    main()
