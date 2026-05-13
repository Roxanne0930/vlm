# fix_accuracy.py
import json

FILE = "outputs/csvqa_results.json"  # 改成你要修正的文件

with open(FILE, "r", encoding="utf-8") as f:
    results = json.load(f)

# 重新统计
correct = sum(1 for s in results["per_sample"] if s["correct"])
total = len(results["per_sample"])
results["correct"] = correct
results["accuracy"] = correct / total if total > 0 else 0

print(f"修正后准确率: {correct}/{total} = {results['accuracy']:.2%}")

with open(FILE, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)