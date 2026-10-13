"""
evaluate.py
===========
Performs in-depth evaluation of trained models:
- Generates full scikit-learn classification reports
- Plots formatted Confusion Matrices with True/False Positive/Negative counts
- Performs Error Analysis on False Positives and False Negatives
- Saves confusion matrix figures to visualizations/
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

PROCESSED_DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "processed", "spam_cleaned.csv"
)
MODELS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"
)
VIS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "visualizations"
)


def evaluate_models():
    """Generates detailed reports, confusion matrix heatmaps, and error analysis."""
    os.makedirs(VIS_DIR, exist_ok=True)

    df = pd.read_csv(PROCESSED_DATA_PATH)
    df["cleaned_message"] = df["cleaned_message"].fillna("")

    X = df["cleaned_message"]
    y = df["label_num"]

    # Use same split as training
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Load production pipeline
    pipeline_path = os.path.join(MODELS_DIR, "spam_classifier_pipeline.joblib")
    if not os.path.exists(pipeline_path):
        raise FileNotFoundError(f"Pipeline not found at: {pipeline_path}. Run train.py first.")

    pipeline = joblib.load(pipeline_path)
    y_pred = pipeline.predict(X_test)

    print("=" * 65)
    print("             PRODUCTION PIPELINE EVALUATION REPORT")
    print("=" * 65)
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Ham (0)", "Spam (1)"], digits=4))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print("Confusion Matrix Breakdown:")
    print(f"  • True Negatives  (Ham correctly identified as Ham)  : {tn}")
    print(f"  • False Positives (Ham incorrectly flagged as Spam)  : {fp} (Critical metric - minimized!)")
    print(f"  • False Negatives (Spam missed and went to Inbox)     : {fn}")
    print(f"  • True Positives  (Spam correctly caught)            : {tp}")

    # ---------------------------------------------------------
    # Plot 1: Best Model Confusion Matrix
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 6))
    group_names = ["True Negative\n(Legit Ham)", "False Positive\n(False Alarm)",
                   "False Negative\n(Missed Spam)", "True Positive\n(Caught Spam)"]
    group_counts = [f"{value:,}" for value in cm.flatten()]
    group_percentages = [f"{value:.2%}" for value in cm.flatten() / np.sum(cm)]
    labels = [f"{v1}\n{v2}\n({v3})" for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2, 2)

    sns.heatmap(cm, annot=labels, fmt="", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["Predicted Ham", "Predicted Spam"],
                yticklabels=["Actual Ham", "Actual Spam"],
                annot_kws={"fontsize": 11, "fontweight": "bold"})
    ax.set_title("Confusion Matrix - Best Spam Classifier (Multinomial NB)", fontweight="bold", pad=12)
    plt.tight_layout()
    cm_path = os.path.join(VIS_DIR, "confusion_matrix_best_model.png")
    fig.savefig(cm_path, dpi=300)
    plt.close(fig)
    print(f"\n[SAVED] Best model confusion matrix: {cm_path}")

    # ---------------------------------------------------------
    # Plot 2: Multi-Model Confusion Matrix Grid (from saved JSON)
    # ---------------------------------------------------------
    metrics_json_path = os.path.join(MODELS_DIR, "model_metrics.json")
    if os.path.exists(metrics_json_path):
        with open(metrics_json_path, "r") as f:
            metadata = json.load(f)

        models_data = metadata["models_evaluated"]
        n_models = len(models_data)
        fig, axes = plt.subplots(1, n_models, figsize=(4.5 * n_models, 4))

        for idx, (m_name, m_info) in enumerate(models_data.items()):
            cur_cm = np.array(m_info["Confusion_Matrix"])
            ax = axes[idx]
            sns.heatmap(cur_cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                        xticklabels=["Ham", "Spam"], yticklabels=["Ham", "Spam"])
            ax.set_title(f"{m_name}\nAcc: {m_info['Accuracy']*100:.1f}% | Prec: {m_info['Precision']*100:.1f}%", fontsize=10, fontweight="bold")
            ax.set_xlabel("Predicted")
            if idx == 0:
                ax.set_ylabel("Actual")
            else:
                ax.set_ylabel("")

        plt.suptitle("Confusion Matrix Comparison Across Models", fontsize=14, fontweight="bold", y=1.03)
        plt.tight_layout()
        grid_cm_path = os.path.join(VIS_DIR, "confusion_matrix_grid.png")
        fig.savefig(grid_cm_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"[SAVED] Confusion matrix grid: {grid_cm_path}")

    # ---------------------------------------------------------
    # Error Analysis: Inspect False Positives & False Negatives
    # ---------------------------------------------------------
    test_indices = X_test.index
    raw_df = pd.read_csv(PROCESSED_DATA_PATH).loc[test_indices].copy()
    raw_df["predicted"] = y_pred

    false_positives = raw_df[(raw_df["label_num"] == 0) & (raw_df["predicted"] == 1)]
    false_negatives = raw_df[(raw_df["label_num"] == 1) & (raw_df["predicted"] == 0)]

    print("\n" + "=" * 65)
    print("                     ERROR ANALYSIS")
    print("=" * 65)
    print(f"Total False Positives (Ham classified as Spam): {len(false_positives)}")
    for i, (_, row) in enumerate(false_positives.iterrows(), 1):
        print(f"  [{i}] Original: {row['message']}")
        print(f"      Cleaned : {row['cleaned_message']}\n")

    print(f"Total False Negatives (Spam classified as Ham): {len(false_negatives)}")
    for i, (_, row) in enumerate(false_negatives.head(3).iterrows(), 1):
        print(f"  [{i}] Original: {row['message']}")
        print(f"      Cleaned : {row['cleaned_message']}\n")


if __name__ == "__main__":
    evaluate_models()
