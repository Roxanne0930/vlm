# check_dureader.py
import json

with open("data/dureader_vis/docvqa_dev.json", "r", encoding="utf-8") as f:
    first_char = f.read(1)
    f.seek(0)
    
    if first_char == '[':
        # 标准 JSON 数组
        data = json.load(f)
        print(f"格式: JSON 数组")
    elif first_char == '{':
        # 可能是 JSONL（每行一个 JSON）或单个对象
        f.seek(0)
        first_line = f.readline().strip()
        try:
            item = json.loads(first_line)
            print(f"格式: JSONL（每行一个 JSON 对象）")
            print(f"第一条示例:")
            for k, v in item.items():
                val_str = str(v)[:200]
                print(f"  {k}: {val_str}")
            
            # 统计总行数
            f.seek(0)
            total = sum(1 for line in f if line.strip())
            print(f"\n总行数: {total}")
            
        except json.JSONDecodeError:
            # 单个 JSON 对象
            f.seek(0)
            data = json.load(f)
            print(f"格式: 单个 JSON 对象")
            print(f"顶层键: {list(data.keys())[:10]}")
    else:
        print(f"未知格式，前100字符: {first_char}")