"""
eda.py
======
Performs comprehensive Exploratory Data Analysis (EDA) on the Spam dataset.
Generates statistical summaries and saves publication-quality visualizations.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Set consistent, modern aesthetic style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 16
})

PROCESSED_DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "processed", "spam_cleaned.csv"
)
VIS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "visualizations"
)


def run_eda(csv_path: str = PROCESSED_DATA_PATH, output_dir: str = VIS_DIR) -> dict:
    """
    Runs complete exploratory data analysis, prints summary stats,
    and produces high-resolution figures.
    """
    os.makedirs(output_dir, exist_ok=True)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Processed file not found: {csv_path}. Run preprocess.py first.")

    df = pd.read_csv(csv_path)

    # 1. Dataset Dimensions & Basic Stats
    n_rows, n_cols = df.shape
    ham_count = (df["label"] == "ham").sum()
    spam_count = (df["label"] == "spam").sum()
    ham_pct = (ham_count / n_rows) * 100
    spam_pct = (spam_count / n_rows) * 100

    print("=" * 60)
    print("        EXPLORATORY DATA ANALYSIS (EDA) REPORT")
    print("=" * 60)
    print(f"Total Rows (Cleaned): {n_rows}")
    print(f"Total Columns: {n_cols}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nTarget Distribution:")
    print(f"  • Ham  (Legitimate): {ham_count:,} ({ham_pct:.2f}%)")
    print(f"  • Spam (Unwanted)  : {spam_count:,} ({spam_pct:.2f}%)")
    print(f"  • Imbalance Ratio  : ~{ham_count / spam_count:.1f}:1 (Ham to Spam)")

    # 2. Length Statistics: Character Count & Word Count
    ham_chars = df[df["label"] == "ham"]["char_count"]
    spam_chars = df[df["label"] == "spam"]["char_count"]
    ham_words = df[df["label"] == "ham"]["word_count"]
    spam_words = df[df["label"] == "spam"]["word_count"]

    stats_summary = {
        "Metric": [
            "Mean Char Count", "Median Char Count", "Std Char Count", "Max Char Count",
            "Mean Word Count", "Median Word Count", "Std Word Count", "Max Word Count"
        ],
        "Ham (Legitimate)": [
            f"{ham_chars.mean():.2f}", f"{ham_chars.median():.1f}", f"{ham_chars.std():.2f}", f"{ham_chars.max()}",
            f"{ham_words.mean():.2f}", f"{ham_words.median():.1f}", f"{ham_words.std():.2f}", f"{ham_words.max()}"
        ],
        "Spam (Unwanted)": [
            f"{spam_chars.mean():.2f}", f"{spam_chars.median():.1f}", f"{spam_chars.std():.2f}", f"{spam_chars.max()}",
            f"{spam_words.mean():.2f}", f"{spam_words.median():.1f}", f"{spam_words.std():.2f}", f"{spam_words.max()}"
        ]
    }
    stats_df = pd.DataFrame(stats_summary)
    print("\nText Length Descriptive Statistics:")
    print(stats_df.to_string(index=False))

    # 3. Sample Messages
    print("\nSample Ham Messages:")
    for i, msg in enumerate(df[df["label"] == "ham"]["message"].head(3), 1):
        print(f"  [{i}] {msg[:100]}...")
    print("\nSample Spam Messages:")
    for i, msg in enumerate(df[df["label"] == "spam"]["message"].head(3), 1):
        print(f"  [{i}] {msg[:100]}...")

    # -------------------------------------------------------------
    # Visualization 1: Class Distribution (Count Bar + Pie Chart)
    # -------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    colors = ["#2b5c8f", "#d9534f"]
    bars = ax1.bar(["Ham (Legitimate)", "Spam"], [ham_count, spam_count], color=colors, edgecolor="black", width=0.55)
    ax1.set_title("Class Frequency Distribution", fontweight="bold")
    ax1.set_ylabel("Message Count")
    for bar in bars:
        h = bar.get_height()
        ax1.annotate(f"{h:,}", xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontweight="bold")

    ax2.pie([ham_count, spam_count], labels=["Ham", "Spam"], autopct="%1.1f%%",
            startangle=140, colors=colors, explode=(0, 0.08), shadow=True,
            textprops={"fontsize": 12, "fontweight": "bold", "color": "white"})
    ax2.set_title("Class Proportions (Imbalance: ~87% vs ~13%)", fontweight="bold")
    plt.tight_layout()
    class_dist_path = os.path.join(output_dir, "class_distribution.png")
    fig.savefig(class_dist_path, dpi=300)
    plt.close(fig)
    print(f"\n[SAVED] Class distribution chart: {class_dist_path}")

    # -------------------------------------------------------------
    # Visualization 2: Character Length Distribution (Histogram + KDE)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.histplot(ham_chars, color="#2b5c8f", label="Ham (Mean: ~71 chars)", kde=True, bins=50, stat="density", alpha=0.5, ax=ax)
    sns.histplot(spam_chars, color="#d9534f", label="Spam (Mean: ~138 chars)", kde=True, bins=50, stat="density", alpha=0.5, ax=ax)
    ax.set_xlim(0, 350)
    ax.set_title("Character Length Distribution: Ham vs Spam", fontweight="bold")
    ax.set_xlabel("Number of Characters per Message")
    ax.set_ylabel("Density")
    ax.legend(frameon=True, facecolor="white", edgecolor="none")
    plt.tight_layout()
    char_len_path = os.path.join(output_dir, "message_length_distribution.png")
    fig.savefig(char_len_path, dpi=300)
    plt.close(fig)
    print(f"[SAVED] Message length chart: {char_len_path}")

    # -------------------------------------------------------------
    # Visualization 3: Word Count Comparison (Boxplot)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.boxplot(x="label", y="word_count", hue="label", legend=False, data=df, palette=colors, ax=ax, width=0.4, showmeans=True,
                meanprops={"marker": "o", "markerfacecolor": "yellow", "markeredgecolor": "black"})
    ax.set_title("Word Count by Class (Yellow dot = Mean)", fontweight="bold")
    ax.set_xlabel("Message Class")
    ax.set_ylabel("Number of Words")
    ax.set_ylim(0, 60)
    plt.tight_layout()
    word_box_path = os.path.join(output_dir, "word_count_distribution.png")
    fig.savefig(word_box_path, dpi=300)
    plt.close(fig)
    print(f"[SAVED] Word count distribution chart: {word_box_path}")

    # -------------------------------------------------------------
    # Visualization 4: Top Vocabulary in Spam vs Ham
    # -------------------------------------------------------------
    spam_words_list = " ".join(df[df["label"] == "spam"]["cleaned_message"].dropna()).split()
    ham_words_list = " ".join(df[df["label"] == "ham"]["cleaned_message"].dropna()).split()

    top_spam = Counter(spam_words_list).most_common(12)
    top_ham = Counter(ham_words_list).most_common(12)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    spam_w, spam_c = zip(*top_spam)
    ham_w, ham_c = zip(*top_ham)

    ax1.barh(range(len(top_spam)), spam_c[::-1], color="#d9534f", edgecolor="black")
    ax1.set_yticks(range(len(top_spam)))
    ax1.set_yticklabels(spam_w[::-1])
    ax1.set_title("Top 12 Words in Spam Messages", fontweight="bold")
    ax1.set_xlabel("Frequency")

    ax2.barh(range(len(top_ham)), ham_c[::-1], color="#2b5c8f", edgecolor="black")
    ax2.set_yticks(range(len(top_ham)))
    ax2.set_yticklabels(ham_w[::-1])
    ax2.set_title("Top 12 Words in Ham Messages", fontweight="bold")
    ax2.set_xlabel("Frequency")

    plt.tight_layout()
    top_words_path = os.path.join(output_dir, "top_words_comparison.png")
    fig.savefig(top_words_path, dpi=300)
    plt.close(fig)
    print(f"[SAVED] Top words comparison chart: {top_words_path}")

    return {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "ham_count": ham_count,
        "spam_count": spam_count,
        "ham_pct": ham_pct,
        "spam_pct": spam_pct,
        "stats_df": stats_df,
    }


if __name__ == "__main__":
    run_eda()
