
import gradio as gr
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification
import torch
import csv
import os
from datetime import datetime

# 加载模型和分词器
model = XLMRobertaForSequenceClassification.from_pretrained("momoali23/finetuned-xlm-esaf-v4")
tokenizer = XLMRobertaTokenizer.from_pretrained("momoali23/finetuned-xlm-esaf-v4")

labels = ["✅ Normal", "🚨 Hate Speech"]
feedback_file = "feedback_log.csv"

def detect_hate(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1).item()
    english_summary = "This text is likely hate speech." if prediction == 1 else "This text is likely safe."
    return labels[prediction], english_summary

def detect_hate_from_file(file):
    if file is None:
        return "❌ No file uploaded.", "Please upload a .txt or .csv file."
    try:
        content = file.read().decode("utf-8")
        return detect_hate(content)
    except Exception as e:
        return "⚠️ Error reading file.", str(e)

# 自定义反馈类
class CustomFlaggingCallback(gr.FlaggingCallback):
    def setup(self, interface, flagging_dir):
        self.interface = interface

    def flag(self, data, flag_option):
        text = data["input"]
        prediction, summary = data["output"]
        if not os.path.exists(feedback_file):
            with open(feedback_file, "w", newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "text", "prediction", "summary", "feedback"])
        with open(feedback_file, "a", newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([datetime.now().isoformat(), text, prediction, summary, flag_option])
        return "Feedback recorded."

# ========== 使用 Blocks 构建界面并自定义输出标签 ==========
with gr.Blocks() as text_input_interface:
    gr.Markdown("### 🌍 Multilingual Hate Speech Detector (Text Input)")
    with gr.Row():
        input_text = gr.Textbox(lines=5, label="Enter text in any of the 6 UN languages:")
    with gr.Row():
        result = gr.Text(label="Result")
        description = gr.Text(label="Description")
    with gr.Row():
        submit_btn = gr.Button("Detect")

    submit_btn.click(fn=detect_hate, inputs=input_text, outputs=[result, description])

    gr.Examples(
        examples=[
            "I love you.",
            "Go back to your country.",
            "You are a kind person.",
            "All [group] are stupid.",
        ],
        inputs=input_text
    )

    # 添加 flag 按钮支持
    gr.Button("Flag", variant="stop").click(
        fn=lambda text, res, desc: CustomFlaggingCallback().flag(
            {"input": text, "output": (res, desc)}, "flagged"
        ),
        inputs=[input_text, result, description],
        outputs=[]
    )

# 上传文件 tab
with gr.Blocks() as file_input_interface:
    gr.Markdown("### 📂 File-based Hate Speech Detection")
    file_input = gr.File(label="Upload a .txt or .csv file", type="binary")
    file_result = gr.Text(label="Result")
    file_desc = gr.Text(label="Description")
    file_btn = gr.Button("Analyze File")

    file_btn.click(fn=detect_hate_from_file, inputs=file_input, outputs=[file_result, file_desc])

# 多标签界面
demo = gr.TabbedInterface(
    interface_list=[text_input_interface, file_input_interface],
    tab_names=["📝 Text Input", "📁 Upload File"]
)

demo.launch()

