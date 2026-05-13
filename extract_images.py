# extract_images.py
import pandas as pd
import os
import base64
import json

PARQUET_FILE = "data/chinese_simple_vqa/chinese_simplevqa.parquet"
OUTPUT_DIR = "data/chinese_simple_vqa/images"
OUTPUT_JSONL = "data/chinese_simple_vqa/ChineseSimpleVQA_local.jsonl"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("读取 parquet 文件...")
df = pd.read_parquet(PARQUET_FILE)

print(f"共 {len(df)} 条数据")
print(f"列名: {list(df.columns)}")

new_data = []
success = 0

for i, row in df.iterrows():
    row_dict = row.to_dict()
    img_id = row_dict.get("ID", f"img_{i}")
    
    # 从 image_base64 解码图片
    image_base64 = row_dict.get("image_base64", "")
    
    if image_base64 and isinstance(image_base64, str) and len(image_base64) > 100:
        try:
            image_bytes = base64.b64decode(image_base64)
            img_filename = f"{img_id}.jpg"
            img_path = os.path.join(OUTPUT_DIR, img_filename)
            with open(img_path, "wb") as f:
                f.write(image_bytes)
            row_dict["image"] = img_filename
            success += 1
        except Exception as e:
            print(f"警告: 第{i}条解码失败: {e}")
            row_dict["image"] = row_dict.get("image_url", "")
    else:
        row_dict["image"] = row_dict.get("image_url", "")
    
    new_data.append(row_dict)

print(f"成功提取: {success}/{len(df)} 张图片")

with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
    for item in new_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"本地 jsonl 已保存: {OUTPUT_JSONL}")