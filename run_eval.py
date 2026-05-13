# run_eval.py
import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from model.vlm_model import VLMQA
from data.dataset_loader import VQADataset
from eval.metrics import evaluate_dataset

API_KEY = "sk-cbcc408bb41847cfa7d04e663aa7198b"          
MODEL_NAME = "qwen3.5-flash"    

print(f"初始化模型: {MODEL_NAME}...")
model = VLMQA(api_key=API_KEY, model_name=MODEL_NAME)

all_results = {}

# ========== 1. ChineseSimpleVQA 公开数据集（100张）==========
print("\n" + "=" * 60)
print("评测1: ChineseSimpleVQA 公开数据集（100张）")
print("=" * 60)
dataset_csvqa = VQADataset("data/chinese_simple_vqa", dataset_type="custom")
dataset_csvqa_subset = dataset_csvqa.get_subset(100)
results_csvqa = evaluate_dataset(
    model, dataset_csvqa_subset,
    output_path="outputs/csvqa_results.json"
)
all_results["ChineseSimpleVQA"] = results_csvqa["accuracy"]
'''
# ========== 2. DuReadervis 文档数据集（30张）==========
print("\n" + "=" * 60)
print("评测2: DuReadervis 文档数据集（30张）")
print("=" * 60)
dataset_dureader = VQADataset("data/dureader_vis", dataset_type="custom")
dataset_dureader_subset = dataset_dureader.get_subset(30)
results_dureader = evaluate_dataset(
    model, dataset_dureader_subset,
    output_path="outputs/dureader_results.json"
)
all_results["DuReadervis"] = results_dureader["accuracy"]

# ========== 3. 自建文档类（15张）==========
print("\n" + "=" * 60)
print("评测3: 自建文档/幻灯片数据集（15张）")
print("=" * 60)
dataset_custom_doc = VQADataset("data/custom_doc", dataset_type="custom")
results_custom_doc = evaluate_dataset(
    model, dataset_custom_doc,
    output_path="outputs/custom_doc_results.json"
)
all_results["自建文档"] = results_custom_doc["accuracy"]

# ========== 4. 自建自然场景类（6张）==========
print("\n" + "=" * 60)
print("评测4: 自建自然场景数据集 (6张) ")
print("=" * 60)
dataset_custom_nat = VQADataset("data/custom_natural", dataset_type="custom")
results_custom_nat = evaluate_dataset(
    model, dataset_custom_nat,
    output_path="outputs/custom_natural_results.json"
)
all_results["自建自然场景"] = results_custom_nat["accuracy"]

# ========== 汇总 ==========
print("\n" + "=" * 60)
print("                    评测汇总")
print("=" * 60)
for name, acc in all_results.items():
    print(f"  {name}: {acc:.2%}")

with open("outputs/summary.json", "w", encoding="utf-8") as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)
print("\n汇总已保存: outputs/summary.json")
'''