import os
import sys
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gradio as gr
import json
from datetime import datetime
from loguru import logger
from PIL import Image
import numpy as np
from src.model.vlm_model import VLMQA  

# ==================== 初始化与目录创建 ====================
os.makedirs("logs", exist_ok=True)
os.makedirs("outputs", exist_ok=True)
os.makedirs("uploads", exist_ok=True)

# ==================== 通用递增文件命名器 ====================
def get_next_filename(folder, prefix="", extension="jpg"):
    today_str = datetime.now().strftime("%Y%m%d")
    existing_files = os.listdir(folder)
    daily_count = 0
    
    match_prefix = f"{prefix}{today_str}"
    for f in existing_files:
        if f.startswith(match_prefix) and f.endswith(f".{extension}"):
            daily_count += 1
            
    next_number = daily_count + 1
    filename = f"{match_prefix}_{next_number:03d}.{extension}"
    
    return os.path.join(folder, filename)


# 配置规范化日志
current_log_path = get_next_filename("logs", prefix="log_", extension="log")
logger.add(current_log_path, rotation="10 MB", retention="7 days")

# 初始化模型
model = VLMQA()


# ==================== 🛠 核心逻辑：智能双模对话函数 ====================
def chat_fn(image, prompt, history, last_image_state, last_path_state):
    """
    增加了两个状态参数：
    - last_image_state: 上一次处理的原始图片数据
    - last_path_state: 上一次图片保存的本地路径
    """
    if not prompt:
        gr.Warning("⚠️ 请输入问题")
        return history, "", last_image_state, last_path_state

    logger.info(f"用户提问：{prompt}")
    
    # 1. 追加用户的文本提问
    history.append({"role": "user", "content": prompt})
    
    # 2. 分流处理：判断是【纯文本模式】还是【多模态图文模式】
    if image is None:
        # ================== 📝 纯文本问答通道 ==================
        logger.info("🔮 检测到纯文本输入，启动文本专用通道...")
        try:
            # 调用你朋友在 vlm_model.py 里写好的纯文本接口
            ans = model.generate_response_text_only(question=prompt)
            logger.success("纯文本模型成功返回数据。")
        except Exception as e:
            ans = f"❌ 纯文本模型调用出错：{str(e)}"
            logger.error(ans)
            
        # 纯文本模式下，清空上一次的图片状态缓存，确保下次传图能正常识别
        last_image_state = None
        final_img_path = None
    else:
        # ================== 🖼️ 多模态图文问答通道 ==================
        # 智能检测：判断当前图片是否和上一轮对话的图片完全一致
        is_same_image = False
        if last_image_state is not None and last_path_state is not None:
            # 如果是 PIL 图像，通过转成 numpy 数组来快速比对像素内容是否完全相同
            img_arr1 = np.array(image)
            img_arr2 = np.array(last_image_state)
            if img_arr1.shape == img_arr2.shape and np.array_equal(img_arr1, img_arr2):
                is_same_image = True

        # 根据比对结果决定是否需要全新缓存
        if is_same_image:
            # 🎯 同一张图连续提问：复用旧路径，不写入硬盘！
            final_img_path = last_path_state
            logger.info(f"🔄 检测到同一张图片的连续提问，直接复用缓存路径：{final_img_path}")
        else:
            # 📸 传入了新图片：生成新编号并固化保存
            final_img_path = get_next_filename("uploads", prefix="", extension="jpg")
            try:
                if isinstance(image, Image.Image):
                    image.save(final_img_path, "JPEG")
                else:
                    Image.fromarray(image).save(final_img_path, "JPEG")
                logger.success(f"📥 收到新图片，已成功保存至新规范路径：{final_img_path}")
            except Exception as e:
                logger.error(f"图片保存失败: {str(e)}")
                final_img_path = None

        # 调用模型接口进行推理
        if final_img_path and os.path.exists(final_img_path):
            try:
                ans = model.generate_response(image_path=final_img_path, question=prompt)
                logger.success(f"模型成功返回：{ans[:50]}...")
            except Exception as e:
                ans = f"❌ 模型调用出错：{str(e)}"
                logger.error(ans)
        else:
            ans = "❌ 图片路径异常，无法调用模型"

        # 更新图片对象状态，留给下一轮对比
        last_image_state = image

    # 3. 追加模型的回答
    history.append({"role": "assistant", "content": ans})
    
    # 4. 返回时，把当前的图片对象和路径塞回 State 里
    return history, "", last_image_state, final_img_path


# ==================== 清空 & 保存对话 ====================
def clear_fn():
    # 清空对话时，同时把图片状态也重置掉
    return [], None, None

def save_fn(history):
    if not history:
        return "无对话可保存"
    fname = f"outputs/chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(fname, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    return f"✅ 保存成功：{fname}"


# ==================== Gradio UI ====================
with gr.Blocks(title="VLM图文助手") as demo:
    gr.Markdown("# 🖼️ VLM 多模态图文问答助手")
    
    # 💡 声明两个不可见的临时状态变量，用来暗中观察图片的变化
    last_image_state = gr.State(None)
    last_path_state = gr.State(None)
    
    with gr.Row():
        with gr.Column(scale=1):
            img = gr.Image(
                type="pil", 
                label="图片输入区",
                sources=["upload", "webcam", "clipboard"]
            )
            prompt = gr.Textbox(label="输入问题", placeholder="输入问题（传图或不传图皆可）...")
            submit = gr.Button("🚀 提交", variant="primary")
            clear_btn = gr.Button("🧹 清空")
            save_btn = gr.Button("💾 保存对话")
            save_info = gr.Textbox(label="保存状态", interactive=False)

        with gr.Column(scale=2):
            chatbot = gr.Chatbot(
                label="对话历史",
                height=650
            )

    # 事件绑定 (将 State 作为输入和输出无缝接轨)
    submit.click(
        chat_fn, 
        inputs=[img, prompt, chatbot, last_image_state, last_path_state], 
        outputs=[chatbot, prompt, last_image_state, last_path_state]
    )
    prompt.submit(
        chat_fn, 
        inputs=[img, prompt, chatbot, last_image_state, last_path_state], 
        outputs=[chatbot, prompt, last_image_state, last_path_state]
    )
    
    # 点击清空按钮时，一并重置聊天框、原始图片缓存状态和路径缓存状态
    clear_btn.click(clear_fn, None, [chatbot, last_image_state, last_path_state])
    save_btn.click(save_fn, [chatbot], save_info)

# ==================== 启动 ====================
if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        debug=True,
        theme=gr.themes.Soft()
    )