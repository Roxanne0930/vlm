# convert_dureader.py
import json
import os

INPUT_FILE = "data/dureader_vis/docvqa_dev.json"
IMAGE_DIR = "data/dureader_vis/dureader_images_dev"
OUTPUT_FILE = "data/dureader_vis/questions.json"

samples = []
skipped = 0

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        
        item = json.loads(line)
        
        image_id = item.get("image_id", "")
        img_filename = f"{image_id}.png"
        img_path = os.path.join(IMAGE_DIR, img_filename)
        
        # 跳过没有本地图片的
        if not os.path.exists(img_path):
            skipped += 1
            continue
        
        question = item.get("question", "")
        answer = item.get("answer", "")
        
        if not question or not answer:
            skipped += 1
            continue
        
        # answer 可能是列表，统一转成列表
        if isinstance(answer, str):
            answers = [answer]
        else:
            answers = answer
        
        # 判断问题类型
        answer_type = item.get("answer_type", "general")
        
        samples.append({
            "id": item.get("id", ""),
            "image": img_path,
            "question": question,
            "answers": answers,
            "question_type": answer_type
        })

print(f"有效样本: {len(samples)}, 跳过: {skipped}")

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(samples, f, ensure_ascii=False, indent=2)

print(f"已保存: {OUTPUT_FILE}")