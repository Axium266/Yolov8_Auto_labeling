import os
from ultralytics import YOLO

def run_auto_annotation(weight_path, image_folder, output_folder, threshold_settings, few_max, medium_max):
    """
    Args:
        weight_path (str): yolov8模型權重路徑 (.pt)
        image_folder (str): 輸入圖片資料夾路徑
        output_folder (str): 輸出標註資料夾路徑
        threshold_settings (dict): 各區段conf/nms/iou字典
        few_max (int): few的最大標籤數
        medium_max (int): medium的最大標籤數 ( many為>medium_max )
    """
    model = YOLO(weight_path)
    os.makedirs(output_folder, exist_ok=True)

    def iou(box1, box2):
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

    for image_name in os.listdir(image_folder):
        if not image_name.lower().endswith((".jpg", ".png", ".jpeg")):
            continue
        image_path = os.path.join(image_folder, image_name)
        # 第一次推論
        initial_results = model(image_path, conf=0.5, iou=0.4)
        # 統計初步偵測數
        label_count = sum(len(result.boxes) for result in initial_results)
        # 分類
        if label_count < 1:
            category = "none"
        elif 1 <= label_count <= few_max:
            category = "few"
        elif few_max < label_count <= medium_max:
            category = "medium"
        else:
            category = "many"

        # 依分類參數設定門檻
        conf_thres = threshold_settings[category]["CONF_THRESHOLD"]
        nms_thres = threshold_settings[category]["NMS_THRESHOLD"]
        iou_thres = threshold_settings[category]["IOU_THRESHOLD"]

        # 第二次推論
        results = model(image_path, conf=conf_thres, iou=nms_thres)
        annotations = []
        for result in results:
            txt_filename = os.path.splitext(image_name)[0] + ".txt"
            txt_path = os.path.join(output_folder, txt_filename)
            for box in result.boxes:
                cls = int(box.cls)
                conf = float(box.conf)
                x_center, y_center, width, height = box.xywhn[0]
                x_min, y_min = x_center - width / 2, y_center - height / 2
                x_max, y_max = x_center + width / 2, y_center + height / 2
                annotations.append((cls, conf, x_min, y_min, x_max, y_max, x_center, y_center, width, height))
            # 消除重複標註
            annotations = sorted(annotations, key=lambda x: x[1], reverse=True)
            final_annotations = []
            temp_anns = list(annotations)
            while temp_anns:
                best = temp_anns.pop(0)
                final_annotations.append(best)
                temp_anns = [ann for ann in temp_anns if iou(best[2:6], ann[2:6]) < iou_thres]
            # 寫入txt
            with open(txt_path, "w") as f:
                for ann in final_annotations:
                    f.write(f"{ann[0]} {ann[6]} {ann[7]} {ann[8]} {ann[9]}\n")
        print(f"標註完成: {image_name} -> {txt_filename}（類別: {category}）")

    print("所有圖片標註完成！")
