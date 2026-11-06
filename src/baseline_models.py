import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from src.preprocess import build_preprocessing_pipeline

# 1. Đọc dữ liệu (Lưu ý dấu phân cách ';' theo đúng README)
train_df = pd.read_csv("train.csv", sep=";")
test_df = pd.read_csv("test (1).csv", sep=";")

target_col = 'y'
X_train = train_df.drop(columns=[target_col])
y_train = train_df[target_col].apply(lambda x: 1 if str(x).strip().lower() in ['yes', '1', 'true'] else 0)

X_test = test_df.drop(columns=[target_col])
y_test = test_df[target_col].apply(lambda x: 1 if str(x).strip().lower() in ['yes', '1', 'true'] else 0)

# 2. Định nghĩa các mô hình Baseline
models = {
    "Dummy (Most Frequent)": DummyClassifier(strategy="most_frequent"),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42)
}

# 3. Chạy thực nghiệm 2 kịch bản (Có và Không có duration)
scenarios = {
    "Kịch bản 1: CÓ duration": True,
    "Kịch bản 2: KHÔNG có duration (Thực tế)": False
}

for scen_name, inc_dur in scenarios.items():
    print(f"\n================ {scen_name} ================")
    # Xây dựng pipeline với cờ include_duration tương ứng
    pipeline = build_preprocessing_pipeline(include_duration=inc_dur)
    
    # Huấn luyện và biến đổi dữ liệu qua Pipeline
    X_train_trans = pipeline.fit_transform(X_train)
    X_test_trans = pipeline.transform(X_test)
    
    for model_name, model in models.items():
        model.fit(X_train_trans, y_train)
        y_pred = model.predict(X_test_trans)
        y_proba = model.predict_proba(X_test_trans)[:, 1] if hasattr(model, "predict_proba") else [0]*len(y_test)
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_proba)
        
        print(f"-> Model: {model_name}")
        print(f"   Accuracy: {acc:.4f} | F1-Score: {f1:.4f} | ROC-AUC: {roc_auc:.4f}")
