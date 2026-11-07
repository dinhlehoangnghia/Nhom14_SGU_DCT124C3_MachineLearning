# -*- coding: utf-8 -*-
"""
Script Thực nghiệm và Đánh giá Mô hình (notebooks/04_modeling_evaluation.py)
"""

import sys
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Nhập module đánh giá nội bộ
sys.path.append('.')
from src.evaluate import compute_metrics, print_evaluation_log, plot_curves

# 1. Đọc dữ liệu Bank Marketing
df = pd.read_csv('data/bank-full.csv', sep=';')

# Tách biến đặc trưng và nhãn target
X = df.drop(columns=['y'])
y = df['y'].map({'no': 0, 'yes': 1}).values

# 2. Cấu hình Pipeline Tiền xử lý
num_cols = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
cat_nominal = ['job', 'marital', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
cat_ordinal = ['education']
edu_order = [['unknown', 'primary', 'secondary', 'tertiary']]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat_nom', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_nominal),
        ('cat_ord', OrdinalEncoder(categories=edu_order), cat_ordinal)
    ]
)

# 3. Phân chia dữ liệu phân tầng Stratified 5-Fold Cross-Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 4. Thiết lập danh sách các mô hình thực nghiệm
models = {
    'Decision Tree (Default)': DecisionTreeClassifier(random_state=42, max_depth=10),
    'Decision Tree (Weighted)': DecisionTreeClassifier(random_state=42, max_depth=10, class_weight='balanced'),
    'Random Forest (Default)': RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1),
    'Random Forest (Weighted)': RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, class_weight='balanced', n_jobs=-1),
    'XGBoost/GradientBoosting (Default)': HistGradientBoostingClassifier(max_iter=100, random_state=42),
    'XGBoost/GradientBoosting (Weighted)': HistGradientBoostingClassifier(max_iter=100, class_weight='balanced', random_state=42)
}

# 5. Huấn luyện mô hình và dự đoán Out-of-Fold (OOF)
summary_list = []
curves_data = {}

for name, model in models.items():
    pipe = Pipeline([('prep', preprocessor), ('clf', model)])
    
    oof_preds = np.zeros(len(y))
    oof_probs = np.zeros(len(y))
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, y_train = X.iloc[train_idx], y[train_idx]
        X_val, y_val = X.iloc[val_idx], y[val_idx]
        
        pipe.fit(X_train, y_train)
        oof_preds[val_idx] = pipe.predict(X_val)
        oof_probs[val_idx] = pipe.predict_proba(X_val)[:, 1]
        
    metrics = compute_metrics(y, oof_preds, oof_probs, model_name=name)
    print_evaluation_log(metrics)
    summary_list.append(metrics)
    curves_data[name] = {'y_true': y, 'y_prob': oof_probs}

# 6. Xuất bảng tổng hợp kết quả
summary_df = pd.DataFrame(summary_list)
display_cols = ['Model', 'Accuracy', 'Precision (1)', 'Recall (1)', 'F1-Score (1)', 'Macro-F1', 'PR-AUC', 'ROC-AUC']
print("=== BẢNG TỔNG HỢP SO SÁNH KẾT QUẢ THỰC NGHIỆM GIỮA KỲ ===")
print(summary_df[display_cols].to_string(index=False))

# 7. Xuất file đồ thị biểu diễn ROC & PR Curves
os.makedirs('reports', exist_ok=True)
plot_curves(curves_data, save_path='reports/roc_pr_curves.png')