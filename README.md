# AI-labels
For yolov8 labeling

先使用run.py 再使用ui.py啟動UI評分標籤

標註好後將feedback.json中的檔案用xgboost訓練
執行train_preference_model.py訓練後
用ui.py測試模型
