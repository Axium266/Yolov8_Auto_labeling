import os
import json
from pathlib import Path
from ultralytics import YOLO
import cv2

# === 設定參數 ===
IMAGE_DIR = 'images'
OUTPUT_DIR = 'outputs'
MODEL_PATH = 'allin.pt'  # 可改成 yolov8m.pt/yolov8s.pt 看效果

CONF_THRESHOLD = 0.5

# 建立資料夾
os.makedirs(f'{OUTPUT_DIR}/labels', exist_ok=True)
os.makedirs(f'{OUTPUT_DIR}/jsons', exist_ok=True)
os.makedirs(f'{OUTPUT_DIR}/images', exist_ok=True)

# 載入模型
model = YOLO(MODEL_PATH)

# 設定要標註的 COCO 類別
TARGET_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "truck", "bus", "red_light", "green_light", "yellow_light"
]

# COCO 類別對照表
COCO_CLASSES = model.names  
NAME_TO_ID = {v: k for k, v in COCO_CLASSES.items()}

# 處理每張圖片
for img_name in os.listdir(IMAGE_DIR):
    img_path = os.path.join(IMAGE_DIR, img_name)
    if not img_path.lower().endswith(('.jpg', '.png', '.jpeg')):
        continue

    results = model(img_path)[0]

    # 取得原圖，準備畫框
    image = cv2.imread(img_path)
    h, w, _ = image.shape

    # 儲存用的資料
    yolo_lines = []
    json_output = []

    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        if conf < CONF_THRESHOLD:
            continue

        class_name = COCO_CLASSES[cls_id]

        if class_name not in TARGET_CLASSES:
            continue

        # YOLO 格式：class_id x_center y_center width height（都正規化）
        xyxy = box.xyxy[0].tolist()
        x1, y1, x2, y2 = map(int, xyxy)
        x_center = ((x1 + x2) / 2) / w
        y_center = ((y1 + y2) / 2) / h
        width = (x2 - x1) / w
        height = (y2 - y1) / h

        # 加入 YOLO 格式
        yolo_lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

        # 加入 JSON 格式
        json_output.append({
            "class_id": cls_id,
            "class_name": class_name,
            "bbox": [x1, y1, x2, y2],
            "confidence": conf
        })

        # 畫框
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f"{class_name} {conf:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

    # 儲存 YOLO txt
    base_name = Path(img_name).stem
    with open(f"{OUTPUT_DIR}/labels/{base_name}.txt", 'w') as f:
        f.write('\n'.join(yolo_lines))

    # 儲存 JSON
    with open(f"{OUTPUT_DIR}/jsons/{base_name}.json", 'w') as f:
        json.dump(json_output, f, indent=2)

    # 儲存畫好的圖片
    cv2.imwrite(f"{OUTPUT_DIR}/images/{base_name}_annotated.jpg", image)

print("標註完成！")