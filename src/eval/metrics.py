# src/eval/metrics.py

import json
import os
from tqdm import tqdm


def compute_accuracy(prediction, reference_answers):
    """计算单条样本准确率"""
    if not prediction or not reference_answers:
        return False, "no_answer"
    
    pred = prediction.strip().lower()
    
    for ref in reference_answers:
        ref = ref.strip().lower()
        
        # 严格匹配
        if pred == ref:
            return True, "exact_match"
        
        # 包含匹配
        if ref in pred or pred in ref:
            return True, "contains_match"
        
        # 关键词匹配
        ref_words = set(ref.split())
        pred_words = set(pred.split())
        if ref_words & pred_words:
            return True, "keyword_match"
    
    return False, "no_match"


def evaluate_dataset(model, dataset, output_path=None, verbose=True):
    """对数据集进行完整评测"""
    results = {
        "dataset_type": dataset.dataset_type,
        "total_samples": len(dataset),
        "correct": 0,
        "match_types": {"exact_match": 0, "contains_match": 0, "keyword_match": 0},
        "per_question_type": {},
        "per_sample": []
    }
    
    for i in tqdm(range(len(dataset)), desc=f"评测 {dataset.dataset_type}"):
        sample = dataset[i]
        
        try:
            prediction = model.generate_response(
                sample["image_path"], 
                sample["question"]
            )
        except Exception as e:
            prediction = ""
            if verbose:
                print(f"\n样本 {i} 推理失败: {e}")
        
        is_correct, match_type = compute_accuracy(prediction, sample["answers"])
        
        if is_correct:
            results["correct"] += 1
            results["match_types"][match_type] += 1
        
        q_type = sample["question_type"]
        if q_type not in results["per_question_type"]:
            results["per_question_type"][q_type] = {"correct": 0, "total": 0}
        results["per_question_type"][q_type]["total"] += 1
        if is_correct:
            results["per_question_type"][q_type]["correct"] += 1
        
        results["per_sample"].append({
            "id": sample["id"],
            "question": sample["question"],
            "prediction": prediction,
            "reference": sample["answers"],
            "correct": is_correct,
            "match_type": match_type,
            "question_type": q_type
        })
    
    results["accuracy"] = results["correct"] / results["total_samples"] if results["total_samples"] > 0 else 0
    
    for q_type in results["per_question_type"]:
        info = results["per_question_type"][q_type]
        info["accuracy"] = info["correct"] / info["total"] if info["total"] > 0 else 0
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"评测结果已保存至: {output_path}")
    
    print(f"\n{'='*50}")
    print(f"数据集: {dataset.dataset_type}")
    print(f"总样本数: {results['total_samples']}")
    print(f"正确数: {results['correct']}")
    print(f"准确率: {results['accuracy']:.2%}")
    print(f"匹配类型: {results['match_types']}")
    print(f"{'='*50}")
    
    return results