# src/data/dataset_loader.py

import json
import os
from PIL import Image


class VQADataset:
    """
    通用 VQA 数据集加载器
    支持 VQA-v2、自建中文数据集、ChineseSimpleVQA
    """
    
    def __init__(self, data_dir, dataset_type="custom"):
        self.data_dir = data_dir
        self.dataset_type = dataset_type
        self.samples = []
        self._load_data()

    def _load_data(self):
        json_path = os.path.join(self.data_dir, "questions.json")
        
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"找不到数据文件: {json_path}")
        
        with open(json_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        
        for item in raw_data:
            sample = {
                "image_path": item.get("image", ""),   # URL 或本地路径
                "question": item["question"],
                "answers": item["answers"] if isinstance(item["answers"], list) else [item["answers"]],
                "question_type": item.get("question_type", "general"),
                "id": item.get("id", "")
            }
            self.samples.append(sample)
        
        print(f"已加载数据集: {len(self.samples)} 条样本")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_path = sample["image_path"]
        
        # 如果是本地路径，加载图片；否则保留URL
        if image_path.startswith("http://") or image_path.startswith("https://"):
            image = image_path  # 保留 URL 给 API
        elif os.path.exists(image_path):
            image = Image.open(image_path).convert("RGB")
        else:
            image = None
        
        return {
            "image": image,
            "image_path": image_path,
            "question": sample["question"],
            "answers": sample["answers"],
            "question_type": sample["question_type"],
            "id": sample["id"]
        }

    def get_subset(self, size):
        import random
        random.seed(42)
        indices = random.sample(range(len(self)), min(size, len(self)))
        subset = VQADataset.__new__(VQADataset)
        subset.data_dir = self.data_dir
        subset.dataset_type = self.dataset_type
        subset.samples = [self.samples[i] for i in indices]
        return subset