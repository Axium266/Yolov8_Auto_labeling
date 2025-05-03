import os
import json
import gradio as gr
import cv2
from PIL import Image
import numpy as np

IMAGE_DIR = "images"
JSON_DIR = "outputs/jsons"
FEEDBACK_FILE = "feedback.json"

# 取得圖片清單
image_list = [f for f in os.listdir(JSON_DIR) if f.endswith('.json')]
current_index = 0
feedback_data = []

# 儲存偏好選擇
def save_feedback(entry):
    data = {}

    if os.path.exists(FEEDBACK_FILE):
        try:
            with open(FEEDBACK_FILE, 'r') as f:
                content = f.read().strip()
                if content:
                    for item in json.loads(content):
                        key = (item['image'], item['class_name'], tuple(item['bbox']))
                        data[key] = item
        except json.JSONDecodeError:
            print("警告：feedback.json 內容錯誤，將覆蓋重寫")
            data = {}

    # 加入或覆蓋現有的標註
    key = (entry['image'], entry['class_name'], tuple(entry['bbox']))
    data[key] = entry

    # 存成 list
    with open(FEEDBACK_FILE, 'w') as f:
        json.dump(list(data.values()), f, indent=2)

# 顯示圖片 + 框，並更新 UI
def load_image_with_boxes(index):
    if index >= len(image_list):
        return None, "所有圖片已完成標記", gr.update(visible=False)

    json_file = image_list[index]
    base_name = os.path.splitext(json_file)[0]
    image_path = os.path.join(IMAGE_DIR, base_name + ".jpg")
    json_path = os.path.join(JSON_DIR, json_file)

    with open(json_path, 'r') as f:
        anns = json.load(f)

    # 讀圖片並畫框
    image = cv2.imread(image_path)
    for ann in anns:
        x1, y1, x2, y2 = ann['bbox']
        class_name = ann['class_name']
        conf = ann['confidence']
        label = f"{class_name} {conf:.2f}"
        cv2.rectangle(image, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(image, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 1)

    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)

    # 傳框資訊回介面
    boxes = [f"{i}: {ann['class_name']} ({ann['confidence']:.2f})" for i, ann in enumerate(anns)]
    return pil_img, f"圖片：{base_name}", gr.update(choices=boxes, value=None)

# 處理喜好按鈕
def mark_preference(index, selected_box, preference):
    json_file = image_list[index]
    json_path = os.path.join(JSON_DIR, json_file)
    with open(json_path, 'r') as f:
        anns = json.load(f)

    if selected_box is None:
        return f"請選擇一個框後再標記偏好！", index

    box_idx = int(selected_box.split(":")[0])
    entry = anns[box_idx]
    entry['image'] = json_file.replace('.json', '.jpg')
    entry['preference'] = preference

    save_feedback(entry)

    message = {
        "like": "已標記為 +1",
        "neutral": "已標記為 0",
        "dislike": "已標記為 -1"
    }
    return message[preference], index

# 換下一張圖
def next_image(index):
    index += 1
    return load_image_with_boxes(index) + (index,)

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# 偏好標註")
    image_box = gr.Image(label="圖片", type="pil")
    status = gr.Textbox()
    box_dropdown = gr.Dropdown(choices=[], label="選擇標註框")

    with gr.Row():
        like_btn = gr.Button(" +1 ")
        neutral_btn = gr.Button(" 0 ")
        dislike_btn = gr.Button(" -1 ")
        next_btn = gr.Button(" 下一張 ")

    result = gr.Textbox(label="標記狀態")
    index_state = gr.State(0)

    # 初始化載入
    demo.load(load_image_with_boxes, inputs=index_state, outputs=[image_box, status, box_dropdown])
    
    # 事件綁定
    like_btn.click(mark_preference, [index_state, box_dropdown, gr.State("like")], result)
    neutral_btn.click(mark_preference, [index_state, box_dropdown, gr.State("neutral")], result)
    dislike_btn.click(mark_preference, [index_state, box_dropdown, gr.State("dislike")], result)
    next_btn.click(next_image, [index_state], [image_box, status, box_dropdown, index_state])

demo.launch()