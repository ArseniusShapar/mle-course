import matplotlib.pyplot as plt
from matplotlib.figure import Figure
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


def calculate_metrics(y_true, y_pred) -> dict[str, float]:
    metrics = (accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, balanced_accuracy_score)
    metrics_scores = {}
    for metric in metrics:
        metrics_scores[metric.__name__] = metric(y_true, y_pred)
    return metrics_scores


def metrics_barplot(y_true, y_pred) -> Figure:
    metrics_scores = calculate_metrics(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
    ax.bar(metrics_scores.keys(), metrics_scores.values())
    ax.set_title("Metric Values")
    ax.set_xticklabels(metrics_scores.keys(), rotation="vertical")
    fig.tight_layout()

    return fig


def build_roc(y_true, y_pred) -> Figure:
    fpr, tpr, _ = roc_curve(y_true, y_pred, pos_label=1)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    ax.plot(fpr, tpr)
    ax.plot([0, 1], [0, 1], "--")
    ax.set_title("ROC Curve")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.grid(True)

    return fig


def build_prc(y_true, y_pred) -> Figure:
    precision, recall, _ = precision_recall_curve(y_true, y_pred, pos_label=1)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    ax.plot(recall, precision)
    ax.set_title("Precision-Recall Curve")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.grid(True)

    return fig
