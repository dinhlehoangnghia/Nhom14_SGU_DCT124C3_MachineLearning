import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin

class BankDataPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, include_duration=True):
        self.include_duration = include_duration
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        X = X.copy()
        
        # 1. Chuyển đổi chuỗi 'unknown' thành NaN
        X.replace('unknown', np.nan, inplace=True)
        
        # 2. Xử lý chuyên sâu cho biến 'pdays'
        if 'pdays' in X.columns:
            # Tạo biến cờ: 1 nếu đã từng liên hệ trước đó, 0 nếu chưa (pdays == 999)
            X['pdays_contacted'] = (X['pdays'] != 999).astype(int)
            # Thay thế giá trị 999 thành NaN để đưa vào imputer/scaler
            X['pdays'] = X['pdays'].replace({999: np.nan, -1: np.nan})
            
        # 3. Loại bỏ duration nếu kịch bản thực tế không cho phép
        if not self.include_duration and 'duration' in X.columns:
            X = X.drop(columns=['duration'])
            
        return X

def build_preprocessing_pipeline(include_duration=True):
    """
    Xây dựng Pipeline hoàn chỉnh gồm Custom Preprocessor và ColumnTransformer.
    """
    # Các cột số (Numerical features)
    numeric_features = ['age', 'campaign', 'pdays', 'previous', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']
    if include_duration:
        numeric_features.append('duration')
        
    # Các cột phân loại (Categorical features)
    categorical_features = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome']

    # Pipeline cho biến số: Điền giá trị median + Chuẩn hóa StandardScaler
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Pipeline cho biến phân loại: Điền giá trị 'missing' + Mã hóa OneHotEncoder
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Kết hợp bằng ColumnTransformer
    col_transformer = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    # Đóng gói toàn bộ vào Pipeline tổng
    full_pipeline = Pipeline(steps=[
        ('custom_prep', BankDataPreprocessor(include_duration=include_duration)),
        ('col_transformer', col_transformer)
    ])

    return full_pipeline
