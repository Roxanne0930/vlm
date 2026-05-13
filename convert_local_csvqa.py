# convert_local_csvqa.py
import json

INPUT_FILE = "data/chinese_simple_vqa/ChineseSimpleVQA_local.jsonl"
OUTPUT_FILE = "data/chinese_simple_vqa/questions.json"

samples = []
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line.strip())
        
        # 用本地图片路径替代 URL
        image_path = f"data/chinese_simple_vqa/images/{item.get('image', '')}"
        
        samples.append({
            "id": item.get("ID", ""),
            "image": image_path,  # 本地路径
            "question": item.get("final_question", ""),
            "answers": [item.get("final_answer", "")],
            "question_type": item.get("Topic", "general")
        })

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(samples, f, ensure_ascii=False, indent=2)

print(f"已生成 {len(samples)} 条: {OUTPUT_FILE}")