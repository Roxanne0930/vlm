# quick_test.py
import sys
sys.path.append("src")

from model.vlm_model import VLMQA

# 替换成你的 API Key，或者提前 set DASHSCOPE_API_KEY=sk-xxx
API_KEY = "sk-cbcc408bb41847cfa7d04e663aa7198b"   # ← 这里填你的 Key
MODEL_NAME = "qwen3.6-flash"  # 或 qwen3.5-plus

print("初始化 API 客户端...")
model = VLMQA(api_key=API_KEY, model_name=MODEL_NAME)

# ====== 测试1：纯文本 ======
print("\n" + "=" * 50)
print("测试1：纯文本对话")
print("=" * 50)
q1 = "你好，请用中文回复，简单介绍一下你自己。"
print(f"问: {q1}")
a1 = model.generate_response_text_only(q1)
print(f"答: {a1}")

# ====== 测试2：图文问答 ======
print("\n" + "=" * 50)
print("测试2：图文问答")
print("=" * 50)
image_path = "test_image.jpg"  # ← 改成你的测试图片路径
q2 = "请描述这张图片里的内容"
print(f"图片: {image_path}")
print(f"问: {q2}")
a2 = model.generate_response(image_path, q2)
print(f"答: {a2}")

# ====== 测试3：多轮对话 ======
print("\n" + "=" * 50)
print("测试3：多轮对话")
print("=" * 50)
questions = [
    "这张图片里有什么？",
    "它是什么颜色的？",
    "它看起来适合什么场合使用？"
]
answers = model.multiround_chat(image_path, questions)
for i, (q, a) in enumerate(zip(questions, answers)):
    print(f"[第{i+1}轮] 问: {q}")
    print(f"[第{i+1}轮] 答: {a}")
    print("---")

print("\n全部测试完成！")