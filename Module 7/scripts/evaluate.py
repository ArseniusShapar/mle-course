import json
import os
import sys

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
)


def calculate_metrics(y_true, y_pred) -> dict:
    metrics = (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        matthews_corrcoef,
        balanced_accuracy_score,
    )
    metrics_scores = {}
    for metric in metrics:
        metrics_scores[metric.__name__] = metric(y_true, y_pred)
    return metrics_scores


def metrics_barplot(y_true, y_pred, output_path):
    metrics_scores = calculate_metrics(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
    ax.bar(metrics_scores.keys(), metrics_scores.values())
    ax.set_title("Metric Values")
    ax.set_xticklabels(metrics_scores.keys(), rotation="vertical")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def build_roc(y_true, y_pred, output_path):
    fpr, tpr, _ = roc_curve(y_true, y_pred, pos_label=1)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    ax.plot(fpr, tpr)
    ax.plot([0, 1], [0, 1], "--")
    ax.set_title("ROC Curve")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def build_prc(y_true, y_pred, output_path):
    precision, recall, _ = precision_recall_curve(y_true, y_pred, pos_label=1)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    ax.plot(recall, precision)
    ax.set_title("Precision-Recall Curve")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


if __name__ == "__main__":
    data_path = sys.argv[1]
    model_path = sys.argv[2]
    metrics_path = sys.argv[3]
    output_dir = sys.argv[4]

    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(data_path)

    y = df.iloc[:, -1]
    X = df.iloc[:, :-1]

    model = joblib.load(model_path)

    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)[:, 1]

    metrics = calculate_metrics(y, y_pred)

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    metrics_barplot(y, y_pred, os.path.join(output_dir, "barplot.png"))
    build_roc(y, y_pred_proba, os.path.join(output_dir, "roc_curve.png"))
    build_prc(y, y_pred_proba, os.path.join(output_dir, "prc_curve.png"))
