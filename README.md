、、、
vlm-qa-assistant/
├── src/
│   ├── model/
│   │   └── vlm_model.py      # VLMQA类：API调用、图像预处理、多轮对话
│   ├── data/
│   │   └── dataset_loader.py  # VQADataset类：统一数据集加载接口
│   └── eval/
│       └── metrics.py         # 评测指标：准确率计算、批量评测流程
├── data/
│   ├── chinese_simple_vqa/    # ChineseSimpleVQA 数据集
│   ├── dureader_vis/          # DuReadervis 文档评测数据
│   ├── custom_doc/            # 自建文档/幻灯片数据
│   └── custom_natural/        # 自建自然场景数据
├── outputs/                   # 评测结果JSON文件
├── run_eval.py                # 评测入口脚本
├── quick_test.py              # API连通性测试
└── requirements.txt           # 依赖列表
、、、
