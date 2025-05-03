import os
import json
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder

FEEDBACK_FILE = "feedback.json"
IMAGE_DIR = "images"

def load_feedback_and_features():
    if not os.path.exists(FEEDBACK_FILE):
        raise FileNotFoundError("找不到 feedback.json")

    with open(FEEDBACK_FILE, 'r') as f:
        data = json.load(f)

    features = []
    labels = []
    class_names = [item["class_name"] for item in data]

    # Label encode class_name
    label_encoder = LabelEncoder()
    label_encoder.fit(class_names)

    for item in data:
        image_path = os.path.join(IMAGE_DIR, item['image'])
        if not os.path.exists(image_path):
            continue  # 忽略找不到的圖片

        img = cv2.imread(image_path)
        h_img, w_img = img.shape[:2]

        x1, y1, x2, y2 = item['bbox']
        x = x1 / w_img
        y = y1 / h_img
        w = (x2 - x1) / w_img
        h = (y2 - y1) / h_img

        class_idx = label_encoder.transform([item['class_name']])[0]
        conf = item['confidence']

        feature = [x, y, w, h, class_idx, conf]
        features.append(feature)

        preference = item['preference']
        label = {"dislike": 0, "neutral": 1, "like": 2}[preference]
        labels.append(label)

    return np.array(features), np.array(labels), label_encoder