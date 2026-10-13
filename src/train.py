"""
train.py
========
Trains and evaluates multiple machine learning classifiers for Spam Detection:
1. Multinomial Naive Bayes
2. Logistic Regression
3. Support Vector Machine (SVC)
4. Random Forest Classifier

Includes:
- Stratified Train/Test split
- TF-IDF Feature Extraction with n-grams
- Multi-metric evaluation (Accuracy, Precision, Recall, F1-score)
- Hyperparameter tuning via GridSearchCV
- Scikit-learn Pipeline serialization using joblib
- Comparison bar chart generation
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

PROCESSED_DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "processed", "spam_cleaned.csv"
)
MODELS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"
)
VIS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "visualizations"
)


def load_data(filepath: str = PROCESSED_DATA_PATH):
    """Loads cleaned data for training."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Cleaned dataset not found: {filepath}. Run preprocess.py first.")
    df = pd.read_csv(filepath)
    df["cleaned_message"] = df["cleaned_message"].fillna("")
    return df


def train_and_evaluate_all():
    """
    Executes the training and comparison pipeline.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(VIS_DIR, exist_ok=True)

    df = load_data()
    X = df["cleaned_message"]
    y = df["label_num"]

    print("=" * 70)
    print("               MODEL TRAINING & BENCHMARKING")
    print("=" * 70)
    print(f"Total instances: {len(df)}")
    print(f"Ham instances  : {(y == 0).sum()} ({(y == 0).mean() * 100:.2f}%)")
    print(f"Spam instances : {(y == 1).sum()} ({(y == 1).mean() * 100:.2f}%)")

    # Stratified Train/Test split (80/20)
    # Stratification ensures exact proportion of Ham (87.37%) and Spam (12.63%) in both train and test splits
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"\nTrain set size : {len(X_train)} (Ham: {(y_train == 0).sum()}, Spam: {(y_train == 1).sum()})")
    print(f"Test set size  : {len(X_test)}  (Ham: {(y_test == 0).sum()}, Spam: {(y_test == 1).sum()})")

    # Define candidate model architectures
    classifiers = {
        "Multinomial Naive Bayes": MultinomialNB(alpha=1.0),
        "Logistic Regression": LogisticRegression(C=1.0, max_iter=1000, random_state=42),
        "Support Vector Machine (Linear)": SVC(kernel="linear", C=1.0, probability=True, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    }

    results = {}
    fitted_pipelines = {}

    print("\n--- Training Models using TF-IDF (1-gram & 2-gram, max 3000 features) ---")

    for name, clf in classifiers.items():
        print(f"\n[TRAINING] {name}...")
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=3000, ngram_range=(1, 2), min_df=2)),
            ("classifier", clf)
        ])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        # Probabilities or decision function for ROC-AUC
        if hasattr(pipeline.named_steps["classifier"], "predict_proba"):
            y_proba = pipeline.predict_proba(X_test)[:, 1]
            roc_auc = roc_auc_score(y_test, y_proba)
        else:
            roc_auc = None

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
        rec = recall_score(y_test, y_pred, pos_label=1)
        f1 = f1_score(y_test, y_pred, pos_label=1)
        cm = confusion_matrix(y_test, y_pred).tolist()

        results[name] = {
            "Accuracy": round(float(acc), 4),
            "Precision": round(float(prec), 4),
            "Recall": round(float(rec), 4),
            "F1-Score": round(float(f1), 4),
            "ROC-AUC": round(float(roc_auc), 4) if roc_auc is not None else None,
            "Confusion_Matrix": cm  # [[TN, FP], [FN, TP]]
        }
        fitted_pipelines[name] = pipeline

        print(f"  • Accuracy : {acc * 100:.2f}%")
        print(f"  • Precision: {prec * 100:.2f}% (Crucial to prevent false spam alarms)")
        print(f"  • Recall   : {rec * 100:.2f}%")
        print(f"  • F1-Score : {f1 * 100:.2f}%")
        if roc_auc:
            print(f"  • ROC-AUC  : {roc_auc * 100:.2f}%")
        print(f"  • Confusion Matrix: TN={cm[0][0]}, FP={cm[0][1]}, FN={cm[1][0]}, TP={cm[1][1]}")

    # Summary Comparison Table
    results_df = pd.DataFrame(results).T[["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]]
    print("\n" + "=" * 70)
    print("                    MODEL PERFORMANCE COMPARISON")
    print("=" * 70)
    print(results_df.to_string())

    # -------------------------------------------------------------------
    # Hyperparameter Tuning on Multinomial Naive Bayes & Logistic Regression
    # -------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("             HYPERPARAMETER TUNING (GridSearchCV)")
    print("=" * 70)
    print("[INFO] Tuning Multinomial Naive Bayes smoothing parameter alpha...")

    nb_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2)),
        ("classifier", MultinomialNB())
    ])

    param_grid_nb = {
        "tfidf__max_features": [2000, 3000, 5000],
        "classifier__alpha": [0.05, 0.1, 0.2, 0.5, 1.0],
    }

    grid_search_nb = GridSearchCV(
        nb_pipeline,
        param_grid=param_grid_nb,
        cv=5,
        scoring="f1",
        n_jobs=-1
    )
    grid_search_nb.fit(X_train, y_train)

    best_nb_params = grid_search_nb.best_params_
    best_nb_cv_f1 = grid_search_nb.best_score_
    tuned_nb_model = grid_search_nb.best_estimator_
    y_pred_tuned_nb = tuned_nb_model.predict(X_test)

    tuned_nb_acc = accuracy_score(y_test, y_pred_tuned_nb)
    tuned_nb_prec = precision_score(y_test, y_pred_tuned_nb, pos_label=1)
    tuned_nb_rec = recall_score(y_test, y_pred_tuned_nb, pos_label=1)
    tuned_nb_f1 = f1_score(y_test, y_pred_tuned_nb, pos_label=1)
    tuned_nb_cm = confusion_matrix(y_test, y_pred_tuned_nb).tolist()

    print(f"Best MNB Parameters: {best_nb_params}")
    print(f"Best 5-Fold CV F1-Score: {best_nb_cv_f1 * 100:.2f}%")
    print(f"Tuned MNB Test Results: Acc={tuned_nb_acc*100:.2f}%, Prec={tuned_nb_prec*100:.2f}%, Rec={tuned_nb_rec*100:.2f}%, F1={tuned_nb_f1*100:.2f}%")
    print(f"Tuned MNB Confusion Matrix: TN={tuned_nb_cm[0][0]}, FP={tuned_nb_cm[0][1]}, FN={tuned_nb_cm[1][0]}, TP={tuned_nb_cm[1][1]}")

    results["Tuned Multinomial Naive Bayes"] = {
        "Accuracy": round(float(tuned_nb_acc), 4),
        "Precision": round(float(tuned_nb_prec), 4),
        "Recall": round(float(tuned_nb_rec), 4),
        "F1-Score": round(float(tuned_nb_f1), 4),
        "ROC-AUC": round(float(roc_auc_score(y_test, tuned_nb_model.predict_proba(X_test)[:, 1])), 4),
        "Confusion_Matrix": tuned_nb_cm,
        "Best_Params": best_nb_params
    }

    # Decide Best Overall Model based on F1-Score and Precision
    # We want top precision so legitimate ham never gets flagged as spam
    ranked_models = sorted(
        results.items(),
        key=lambda item: (item[1]["Precision"] * 0.6 + item[1]["F1-Score"] * 0.4),
        reverse=True
    )
    best_model_name = ranked_models[0][0]
    print(f"\n[SELECTION] Best Overall Model Selected: '{best_model_name}'")

    if best_model_name == "Tuned Multinomial Naive Bayes":
        best_pipeline = tuned_nb_model
    else:
        best_pipeline = fitted_pipelines[best_model_name]

    # Save the Pipeline and Vectorizer
    pipeline_save_path = os.path.join(MODELS_DIR, "spam_classifier_pipeline.joblib")
    joblib.dump(best_pipeline, pipeline_save_path)
    print(f"[SAVED] Production Pipeline saved to: {pipeline_save_path}")

    # Also save vectorizer and classifier separately for educational modularity
    vectorizer_save_path = os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib")
    clf_save_path = os.path.join(MODELS_DIR, "best_classifier.joblib")
    joblib.dump(best_pipeline.named_steps["tfidf"], vectorizer_save_path)
    joblib.dump(best_pipeline.named_steps["classifier"], clf_save_path)
    print(f"[SAVED] Modular Vectorizer: {vectorizer_save_path}")
    print(f"[SAVED] Modular Classifier: {clf_save_path}")

    # Save metadata JSON
    metadata = {
        "dataset_statistics": {
            "total_samples": len(df),
            "ham_samples": int((y == 0).sum()),
            "spam_samples": int((y == 1).sum()),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
        },
        "models_evaluated": results,
        "selected_best_model": best_model_name,
        "selected_model_metrics": results[best_model_name],
    }

    metadata_path = os.path.join(MODELS_DIR, "model_metrics.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"[SAVED] Model evaluation metrics JSON: {metadata_path}")

    # -------------------------------------------------------------------
    # Visualization: Model Comparison Bar Chart
    # -------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 6))
    comparison_keys = [m for m in results.keys()]
    metrics_to_plot = ["Accuracy", "Precision", "Recall", "F1-Score"]
    
    x = np.arange(len(comparison_keys))
    width = 0.18
    palette = ["#2b5c8f", "#2ca02c", "#d9534f", "#e67e22"]

    for i, metric in enumerate(metrics_to_plot):
        vals = [results[m][metric] * 100 for m in comparison_keys]
        bars = ax.bar(x + (i - 1.5) * width, vals, width, label=metric, color=palette[i], edgecolor="black")
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                        fontsize=8, fontweight="bold", rotation=45)

    ax.set_title("Machine Learning Models Performance Benchmark", fontweight="bold", pad=15)
    ax.set_ylabel("Score (%)")
    ax.set_ylim(80, 105)
    ax.set_xticks(x)
    ax.set_xticklabels(comparison_keys, rotation=15, ha="right", fontweight="bold")
    ax.legend(loc="lower right", frameon=True, facecolor="white")
    plt.tight_layout()

    comp_chart_path = os.path.join(VIS_DIR, "model_comparison.png")
    fig.savefig(comp_chart_path, dpi=300)
    plt.close(fig)
    print(f"[SAVED] Model comparison chart: {comp_chart_path}")

    return metadata


if __name__ == "__main__":
    train_and_evaluate_all()
