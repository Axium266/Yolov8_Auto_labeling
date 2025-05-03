import xgboost as xgb
import joblib
from train_utils import load_feedback_and_features

# 讀取特徵與標籤
X, y, label_encoder = load_feedback_and_features()

# 訓練 XGBoost 分類模型
model = xgb.XGBClassifier(
    objective="multi:softprob",
    num_class=3,
    eval_metric="mlogloss",
    use_label_encoder=False,
    random_state=42
)
model.fit(X, y)

# 儲存模型與編碼器
joblib.dump(model, "preference_model_xgb.pkl")
joblib.dump(label_encoder, "class_label_encoder.pkl")

print("模型訓練完成並已儲存！")