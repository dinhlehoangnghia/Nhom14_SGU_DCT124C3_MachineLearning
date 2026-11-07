# -*- coding: utf-8 -*-
"""
Module đánh giá mô hình dùng chung (src/evaluate.py)
Đồ án Máy học - Bài toán Bank Marketing
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)


def compute_metrics(y_true, y_pred, y_prob, model_name="Model"):
    """
    Tính toán toàn bộ chỉ số đánh giá cho bài toán Phân loại Nhị phân.
    
    Parameters:
    -----------
    y_true : array-like, nhãn thực tế (0 hoặc 1).
    y_pred : array-like, nhãn dự đoán từ mô hình (0 hoặc 1).
    y_prob : array-like, xác suất dự đoán cho lớp dương (1).
    model_name : str, tên mô hình phục vụ báo cáo.
    
    Returns:
    --------
    dict: Từ điển chứa đầy đủ chỉ số và các phần tử ma trận nhầm lẫn.
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    metrics = {
        "Model": model_name,
        "Accuracy": (tp + tn) / (tp + tn + fp + fn),
        "Precision (1)": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "Recall (1)": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "F1-Score (1)": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "Macro-F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "PR-AUC": average_precision_score(y_true, y_prob),
        "ROC-AUC": roc_auc_score(y_true, y_prob),
        "TN": tn, "FP": fp, "FN": fn, "TP": tp
    }
    return metrics


def print_evaluation_log(metrics):
    """
    In dòng log kết quả thực nghiệm ra Console định dạng chuẩn.
    """
    print(f"==================================================")
    print(f"[EVAL LOG] Model: {metrics['Model']}")
    print(f"--------------------------------------------------")
    print(f" Accuracy      : {metrics['Accuracy']:.4f}")
    print(f" Precision (1) : {metrics['Precision (1)']:.4f}")
    print(f" Recall (1)    : {metrics['Recall (1)']:.4f}")
    print(f" F1-Score (1)  : {metrics['F1-Score (1)']:.4f}")
    print(f" Macro-F1      : {metrics['Macro-F1']:.4f}")
    print(f" PR-AUC        : {metrics['PR-AUC']:.4f}")
    print(f" ROC-AUC       : {metrics['ROC-AUC']:.4f}")
    print(f" Confusion Matrix (TN/FP/FN/TP): {metrics['TN']} / {metrics['FP']} / {metrics['FN']} / {metrics['TP']}")
    print(f"==================================================\n")


def plot_curves(results_dict, save_path=None):
    """
    Vẽ biểu đồ so sánh ROC Curve và Precision-Recall Curve giữa các mô hình.
    
    Parameters:
    -----------
    results_dict : dict
        Dạng {model_name: {'y_true': y_true, 'y_prob': y_prob}}
    save_path : str, optional
        Đường dẫn xuất file ảnh (.png).
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. Receiver Operating Characteristic (ROC) Curve
    for model_name, data in results_dict.items():
        fpr, tpr, _ = roc_curve(data['y_true'], data['y_prob'])
        auc_val = roc_auc_score(data['y_true'], data['y_prob'])
        axes[0].plot(fpr, tpr, lw=2, label=f"{model_name} (AUC = {auc_val:.3f})")
        
    axes[0].plot([0, 1], [0, 1], 'k--', lw=1, label="Random Baseline")
    axes[0].set_title("Biểu đồ đường cong ROC (ROC Curve)", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("False Positive Rate (FPR)")
    axes[0].set_ylabel("True Positive Rate (Recall)")
    axes[0].legend(loc="lower right")
    axes[0].grid(True, alpha=0.3)
    
    # 2. Precision-Recall (PR) Curve
    for model_name, data in results_dict.items():
        prec, rec, _ = precision_recall_curve(data['y_true'], data['y_prob'])
        pr_auc_val = average_precision_score(data['y_true'], data['y_prob'])
        axes[1].plot(rec, prec, lw=2, label=f"{model_name} (PR-AUC = {pr_auc_val:.3f})")
        
    axes[1].set_title("Biểu đồ đường cong Precision-Recall (PR Curve)", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].legend(loc="lower left")
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[INFO] Đã lưu đồ thị ROC/PR Curve tại: {save_path}")
    plt.show()