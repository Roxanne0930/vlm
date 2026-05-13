# src/model/vlm_model.py

import os
import base64
from openai import OpenAI


class VLMQA:
    def __init__(self, api_key=None, model_name="qwen3.5-flash"):
        """
        使用阿里云百炼 DashScope API 调用多模态模型
        - api_key: 不传则从环境变量 DASHSCOPE_API_KEY 读取
        - model_name: 推荐 qwen3.6-flash / qwen3.5-plus
        """
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise ValueError("请设置 DASHSCOPE_API_KEY 环境变量或在初始化时传入 api_key")

        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        self.model_name = model_name
        print(f"API 客户端初始化完成，模型: {model_name}")

    # ==================== 内部方法 ====================

    def _encode_image(self, image_path):
        """将本地图片转为 base64 编码"""
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode("utf-8")

    def _call_api(self, messages):
        """统一调用 API 并返回回答文本"""
        print("  [API 调用中...]", end=" ")
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.1,
                max_tokens=512
            )
            print("完成")
            return completion.choices[0].message.content
        except Exception as e:
            print(f"失败")
            return f"API 调用失败: {str(e)}"

    def _decode(self, inputs, generated_ids):
        """（API 模式保留空方法，兼容后续本地部署代码）"""
        pass

    # ==================== 对外接口 ====================

    def generate_response_text_only(self, question):
        """纯文本问答：快速验证 API 是否正常"""
        messages = [
            {"role": "user", "content": question}
        ]
        return self._call_api(messages)

    def generate_response(self, image_path, question):
        """
        单轮图文问答：支持本地路径和 URL
        """
        # 判断是 URL 还是本地路径
        if image_path.startswith("http://") or image_path.startswith("https://"):
            image_url = image_path
        else:
            image_base64 = self._encode_image(image_path)
            image_url = f"data:image/jpeg;base64,{image_base64}"

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image_url", "image_url": {"url": image_url}}
                ]
            }
        ]
        return self._call_api(messages)

    def multiround_chat(self, image_path, questions):
        """
        多轮对话：一张图片 + 多个连续问题
        - image_path: 本地图片路径
        - questions: 问题列表，如 ["这是什么？", "什么颜色？", "适合什么人？"]
        - 返回: 回答列表，与问题一一对应
        """
        image_base64 = self._encode_image(image_path)
        image_url = f"data:image/jpeg;base64,{image_base64}"
        messages = []
        answers = []

        for i, question in enumerate(questions):
            if i == 0:
                # 第一轮：带图片
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {"type": "image_url", "image_url": {"url": image_url}}
                    ]
                })
            else:
                # 后续轮：纯文本
                messages.append({
                    "role": "user",
                    "content": question
                })

            answer = self._call_api(messages)
            answers.append(answer)
            messages.append({"role": "assistant", "content": answer})

        return answers