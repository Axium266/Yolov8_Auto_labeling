import os
import numpy as np
from ultralytics import YOLO

# YOLO 模型
model1 = YOLO("C:/Users/USER/Desktop/all image/auto/yellow.pt") 
model2 = YOLO("C:/Users/USER/Desktop/all image/auto/all.pt")  

# 設定資料夾
image_folder = "C:/Users/USER/Desktop/all image/auto/images"
output_folder = "C:/Users/USER/Desktop/all image/auto/labels"
os.makedirs(output_folder, exist_ok=True)

# 參數設定
NMS_THRESHOLD = 0.5    # 非極大值抑制 (NMS)
CONF_THRESHOLD = 0.6  # 置信度閾值
IOU_THRESHOLD = 0.2    # IoU 閾值（去除重複標註）

# model1 重新對應標籤
remap_classes_model1 = {0: 6, 1: 7} 

# model2 重新對應標籤
remap_classes_model2 = {0: 0, 1: 1, 2: 2, 3: 3, 7: 4, 5: 5}  
valid_classes_model2 = set(remap_classes_model2.keys())  

# 計算 IoU
def iou(box1, box2):
    """ 計算兩個 bounding box 的 IoU 值 """
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    inter_x1 = max(x1_min, x2_min)
    inter_y1 = max(y1_min, y2_min)
    inter_x2 = min(x1_max, x2_max)
    inter_y2 = min(y1_max, y2_max)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)

    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

# 處理每張圖片
for image_name in os.listdir(image_folder):
    if image_name.endswith((".jpg", ".png", ".jpeg")):
        image_path = os.path.join(image_folder, image_name)

        # 使用模型進行偵測
        results1 = model1(image_path, conf=CONF_THRESHOLD, iou=NMS_THRESHOLD)
        results2 = model2(image_path, conf=CONF_THRESHOLD, iou=NMS_THRESHOLD)

        # 存儲標註
        annotations = []

        # 處理模型1的標註（紅綠燈）
        for result in results1:
            for box in result.boxes:
                cls = int(box.cls)
                cls = remap_classes_model1.get(cls, None)  # 重新映射標籤
                if cls is not None:
                    conf = float(box.conf)
                    x_center, y_center, width, height = box.xywhn[0]
                    x_min, y_min = x_center - width / 2, y_center - height / 2
                    x_max, y_max = x_center + width / 2, y_center + height / 2
                    annotations.append((cls, conf, x_min, y_min, x_max, y_max, x_center, y_center, width, height))

        # 處理模型2的標註（車輛、行人）
        for result in results2:
            for box in result.boxes:
                cls = int(box.cls)
                if cls in valid_classes_model2:
                    cls = remap_classes_model2[cls]  # 重新映射標籤
                    conf = float(box.conf)
                    x_center, y_center, width, height = box.xywhn[0]
                    x_min, y_min = x_center - width / 2, y_center - height / 2
                    x_max, y_max = x_center + width / 2, y_center + height / 2
                    annotations.append((cls, conf, x_min, y_min, x_max, y_max, x_center, y_center, width, height))

        # 根據 IoU 去除重疊標註
        annotations = sorted(annotations, key=lambda x: x[1], reverse=True)  # 根據置信度排序
        final_annotations = []

        while annotations:
            best = annotations.pop(0)  # 取出置信度最高的標註
            final_annotations.append(best)
            # 移除與此標註 IoU 過高的標註
            annotations = [ann for ann in annotations if iou(best[2:6], ann[2:6]) < IOU_THRESHOLD]

        # 儲存 YOLO 格式標註
        txt_filename = os.path.splitext(image_name)[0] + ".txt"
        txt_path = os.path.join(output_folder, txt_filename)

        with open(txt_path, "w") as f:
            for ann in final_annotations:
                f.write(f"{ann[0]} {ann[6]} {ann[7]} {ann[8]} {ann[9]}\n")

        print(f"標註完成: {image_name} -> {txt_filename}")

print("所有圖片標註完成！")