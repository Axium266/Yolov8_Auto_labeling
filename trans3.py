import os
import numpy as np
from ultralytics import YOLO

# 模型權重
model = YOLO("C:/Users/USER/Desktop/test/best.pt")

# 輸入圖片 與 輸出資料夾
image_folder = "C:/Users/USER/Desktop/test/images"
output_folder = "C:/Users/USER/Desktop/test/labels"
os.makedirs(output_folder, exist_ok=True)

# 設定參數
NMS_THRESHOLD = 0.5  
CONF_THRESHOLD = 0.5  
IOU_THRESHOLD = 0.5  

def iou(box1, box2):
    """ 計算兩個 bounding box 的 IoU 值 """
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    # 計算交集
    inter_x1 = max(x1_min, x2_min)
    inter_y1 = max(y1_min, y2_min)
    inter_x2 = min(x1_max, x2_max)
    inter_y2 = min(y1_max, y2_max)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    # 計算每個框的面積
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)

    # 計算 IoU
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

# 批量處理圖片
for image_name in os.listdir(image_folder):
    if image_name.endswith((".jpg", ".png", ".jpeg")):
        image_path = os.path.join(image_folder, image_name)

        # 進行物件偵測
        results = model(image_path, conf=CONF_THRESHOLD, iou=NMS_THRESHOLD)

        # 儲存標註
        annotations = []
        
        for result in results:
            txt_filename = os.path.splitext(image_name)[0] + ".txt"
            txt_path = os.path.join(output_folder, txt_filename)

            for box in result.boxes:
                cls = int(box.cls)  # 類別編號
                conf = float(box.conf)  # 置信度
                x_center, y_center, width, height = box.xywhn[0]  # 轉換成 YOLO 格式 (歸一化)
                
                # 計算左上與右下座標（相對於 1x1 範圍）
                x_min, y_min = x_center - width / 2, y_center - height / 2
                x_max, y_max = x_center + width / 2, y_center + height / 2
                
                annotations.append((cls, conf, x_min, y_min, x_max, y_max, x_center, y_center, width, height))

            # 根據 IoU 過濾重複標註
            annotations = sorted(annotations, key=lambda x: x[1], reverse=True)  # 根據置信度排序
            final_annotations = []

            while annotations:
                best = annotations.pop(0)  # 取出置信度最高的標註
                final_annotations.append(best)

                # 移除與此標註 IoU 過高的標註
                annotations = [ann for ann in annotations if iou(best[2:6], ann[2:6]) < IOU_THRESHOLD]

            # 寫入標註文件
            with open(txt_path, "w") as f:
                for ann in final_annotations:
                    f.write(f"{ann[0]} {ann[6]} {ann[7]} {ann[8]} {ann[9]}\n")

        print(f"標註完成: {image_name} -> {txt_filename}")

print("所有圖片標註完成")