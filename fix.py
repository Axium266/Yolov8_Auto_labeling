import os

# 標註檔案資料夾
labels_dir = "C:/Users/USER/Desktop/all image/auto/labels" 

for file in os.listdir(labels_dir):
    if file.endswith(".txt"):
        file_path = os.path.join(labels_dir, file)
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        fixed_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 5:  # yolo格式
                class_id = parts[0]
                values = [f"{float(x):.6f}" for x in parts[1:]]  # 保留6位小數
                fixed_lines.append(f"{class_id} {' '.join(values)}")

        with open(file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(fixed_lines))
        
        print(f"已修正標註文件: {file_path}")
