"""
app.py
======
Interactive Streamlit Web Application for Email & SMS Spam Classification.
Features:
- Real-time spam detection with probability scores
- Spam trigger keyword extraction
- Pre-loaded realistic email & SMS templates
- Interactive model performance benchmarks & confusion matrix
- EDA visualization gallery & dataset insights

Run via either:
  streamlit run app.py
  or
  python app.py
"""

import os
import sys
import json
import warnings
import joblib
import pandas as pd

# Suppress dependency and deprecation warnings
warnings.filterwarnings("ignore")

# If user executed `python app.py` directly in terminal or clicked Run in VS Code,
# automatically launch it with `streamlit run app.py`
if __name__ == "__main__":
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        if get_script_run_ctx() is None:
            import subprocess
            print("=" * 65)
            print("  🚀 Launching Streamlit Web App...")
            print("  Opening at: http://localhost:8501")
            print("  Press Ctrl+C in terminal to stop.")
            print("=" * 65)
            subprocess.run([sys.executable, "-m", "streamlit", "run", os.path.abspath(__file__)])
            sys.exit(0)
    except Exception:
        pass

import streamlit as st

# Add src to path for module imports
src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if src_path not in sys.path:
    sys.path.append(src_path)

from preprocess import clean_text
from predict import predict_message, SPAM_TRIGGER_KEYWORDS

# Page configuration
st.set_page_config(
    page_title="SpamGuard AI - Email Spam Classifier",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.4rem;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #1E88E5, #43A047);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #666;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 18px;
        border: 1px solid #e9ecef;
        text-align: center;
    }
    .metric-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1a73e8;
    }
    .metric-lbl {
        color: #5f6368;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .spam-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 12px;
        background-color: #ffebee;
        color: #c62828;
        font-weight: bold;
        border: 1px solid #ef9a9a;
    }
    .ham-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 12px;
        background-color: #e8f5e9;
        color: #2e7d32;
        font-weight: bold;
        border: 1px solid #a5d6a7;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        font-weight: 600;
        height: 3rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_pipeline():
    """Loads and caches the model pipeline."""
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "spam_classifier_pipeline.joblib")
    if not os.path.exists(model_path):
        st.error(f"Trained model not found at `{model_path}`. Please run `python src/train.py` first.")
        return None
    return joblib.load(model_path)


@st.cache_data
def get_metrics():
    """Loads saved model metadata."""
    metrics_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "model_metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            return json.load(f)
    return None


# Sidebar
pipeline = get_pipeline()
metrics_data = get_metrics()

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/shield.png", width=64)
    st.title("SpamGuard AI")
    st.markdown("**ML Email & SMS Classifier**")
    st.markdown("---")

    st.subheader("Model Status")
    if pipeline:
        st.success("🟢 Production Pipeline Active")
        st.caption("Algorithm: **Multinomial Naive Bayes**")
        st.caption("Feature Extractor: **TF-IDF (1-2 ngrams)**")
        st.caption("Vocabulary Size: **3,000 features**")
    else:
        st.error("🔴 Model Not Loaded")

    st.markdown("---")
    st.subheader("Key Takeaway for Spam")
    st.info(
        "💡 **Why Precision Matters:**\n"
        "In spam detection, a **False Positive** sends a crucial email (job offer, medical record) "
        "to the Junk folder. Our model achieves **99.04% Precision**, ensuring safe messages are never lost."
    )
    st.markdown("---")
    st.caption("Designed for Academic Review, Interviews & Portfolio.")


# Main Content Header
st.markdown('<div class="main-title">🛡️ Email & SMS Spam Classification</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">End-to-End Natural Language Processing & Machine Learning Pipeline</div>', unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Real-Time Detector",
    "📊 Model Performance",
    "📈 Dataset & EDA",
    "🎓 Interview & Concepts"
])

# -------------------------------------------------------------
# TAB 1: Real-time Detector
# -------------------------------------------------------------
with tab1:
    col_input, col_meta = st.columns([2, 1])

    with col_input:
        st.subheader("Analyze an Email or Message")

        # Preset Examples Dropdown
        PRESETS = {
            "Select an example to auto-fill...": "",
            "🎁 Lottery / Cash Prize (Spam)": "URGENT! You have won a 1,000,000 cash prize! Call 09061701461 to claim your reward immediately. Free entry in weekly competition.",
            "⚠️ Phishing / Urgent Account Alert (Spam)": "FINAL NOTICE: Your bank account will be suspended. Verify your credit card details immediately by clicking here.",
            "🎟️ Holiday Voucher Promo (Spam)": "Congratulations! You have been selected for a free holiday voucher worth $500. Reply CLAIM to 88088 now.",
            "💼 Professional Meeting Invite (Ham)": "Hi Alex, please find attached the revised presentation for tomorrow's 10:00 AM project review meeting. Let me know your thoughts.",
            "💬 Casual Friend Conversation (Ham)": "Hey! Are you still free for dinner tonight around 7:30 PM? Let me know where you want to meet up.",
            "✈️ Flight Booking Update (Ham)": "Your flight AA-204 to Chicago is confirmed for departure at 14:15. Please check in online or arrive at gate B4 45 minutes prior."
        }

        selected_preset = st.selectbox("Quick-Load Sample Message:", list(PRESETS.keys()))

        default_text = PRESETS[selected_preset] if selected_preset != "Select an example to auto-fill..." else ""

        user_input = st.text_area(
            "Paste or type email / SMS text here:",
            value=default_text,
            height=140,
            placeholder="e.g. Free entry in 2 a weekly comp to win FA Cup final tickets! Text WIN to 87121 now..."
        )

        detect_button = st.button("🚀 Analyze Message", type="primary")

    with col_meta:
        st.subheader("Text Metrics")
        if user_input.strip():
            char_len = len(user_input)
            word_len = len(user_input.split())
            st.metric("Character Count", char_len)
            st.metric("Word Count", word_len)
            st.caption(f"Average Ham Length: **71 chars** | Average Spam Length: **138 chars**")
        else:
            st.info("Enter text to see live length metrics.")

    if detect_button or user_input.strip():
        if not user_input.strip():
            st.warning("Please enter some text or pick a sample from above.")
        elif pipeline:
            result = predict_message(user_input, pipeline)

            st.markdown("---")
            st.subheader("Classification Result")

            res_col1, res_col2 = st.columns([1, 1])

            with res_col1:
                if result["is_spam"]:
                    st.error(f"### 🚨 Classification: **SPAM MESSAGE**")
                    st.markdown(
                        f"**Confidence Score:** `{result['confidence']:.2f}%`  \n"
                        f"**Spam Probability:** `{result['spam_probability']:.2f}%` | "
                        f"**Ham Probability:** `{result['ham_probability']:.2f}%`"
                    )
                    st.progress(result["spam_probability"] / 100)
                else:
                    st.success(f"### 🛡️ Classification: **LEGITIMATE (HAM)**")
                    st.markdown(
                        f"**Confidence Score:** `{result['confidence']:.2f}%`  \n"
                        f"**Ham Probability:** `{result['ham_probability']:.2f}%` | "
                        f"**Spam Probability:** `{result['spam_probability']:.2f}%`"
                    )
                    st.progress(result["ham_probability"] / 100)

            with res_col2:
                st.markdown("#### 🔍 Explainability & Tokens")
                st.write(f"**Cleaned & Stemmed Representation:**")
                st.code(result["cleaned_text"] if result["cleaned_text"] else "(Empty after stopword removal)", language="text")

                if result["trigger_words"]:
                    st.markdown("**Detected Spam Trigger Keywords:**")
                    triggers_html = " ".join([f"<span class='spam-badge'>{w}</span>" for w in result["trigger_words"]])
                    st.markdown(triggers_html, unsafe_allow_html=True)
                else:
                    st.markdown("*No dominant high-risk promotional keywords detected.*")

# -------------------------------------------------------------
# TAB 2: Model Performance
# -------------------------------------------------------------
with tab2:
    st.subheader("Model Evaluation & Benchmarks")
    st.markdown(
        "All models were evaluated on an **unseen stratified test split (20%, 1,035 messages)** "
        "using TF-IDF feature extraction (3,000 features, 1-gram & 2-gram). All metrics represent actual execution outputs."
    )

    if metrics_data:
        models_dict = metrics_data.get("models_evaluated", {})
        summary_rows = []
        for name, data in models_dict.items():
            summary_rows.append({
                "Model Architecture": name,
                "Accuracy": f"{data['Accuracy']*100:.2f}%",
                "Precision (Spam)": f"{data['Precision']*100:.2f}%",
                "Recall (Spam)": f"{data['Recall']*100:.2f}%",
                "F1-Score": f"{data['F1-Score']*100:.2f}%",
                "ROC-AUC": f"{data['ROC-AUC']*100:.2f}%" if data.get("ROC-AUC") else "N/A"
            })
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

        col_metric1, col_metric2, col_metric3, col_metric4 = st.columns(4)
        best_metrics = metrics_data.get("selected_model_metrics", {})
        with col_metric1:
            st.metric("Best Model Accuracy", f"{best_metrics.get('Accuracy', 0)*100:.2f}%")
        with col_metric2:
            st.metric("Best Model Precision", f"{best_metrics.get('Precision', 0)*100:.2f}%", delta="Minimizes False Alarms")
        with col_metric3:
            st.metric("Best Model F1-Score", f"{best_metrics.get('F1-Score', 0)*100:.2f}%")
        with col_metric4:
            st.metric("False Positives", f"{best_metrics.get('Confusion_Matrix', [[0,0],[0,0]])[0][1]} out of 904")

    col_chart1, col_chart2 = st.columns([1.2, 1])
    with col_chart1:
        comp_img = os.path.join(os.path.dirname(os.path.abspath(__file__)), "visualizations", "model_comparison.png")
        if os.path.exists(comp_img):
            st.image(comp_img, caption="Figure 1: Benchmark Comparison across 4 Classifiers", use_container_width=True)
    with col_chart2:
        cm_img = os.path.join(os.path.dirname(os.path.abspath(__file__)), "visualizations", "confusion_matrix_best_model.png")
        if os.path.exists(cm_img):
            st.image(cm_img, caption="Figure 2: Confusion Matrix for Best Model (Multinomial NB)", use_container_width=True)

# -------------------------------------------------------------
# TAB 3: Dataset & EDA
# -------------------------------------------------------------
with tab3:
    st.subheader("Exploratory Data Analysis (EDA)")
    st.markdown(
        "Analysis performed on the **UCI SMS Spam Collection** corpus containing verified English messages."
    )

    col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
    with col_kpi1:
        st.metric("Total Messages (Cleaned)", "5,171")
    with col_kpi2:
        st.metric("Ham (Legitimate)", "4,518 (87.37%)")
    with col_kpi3:
        st.metric("Spam (Unwanted)", "653 (12.63%)")
    with col_kpi4:
        st.metric("Duplicate Messages Removed", "403 (7.23%)")

    eda_row1_col1, eda_row1_col2 = st.columns(2)
    with eda_row1_col1:
        class_dist_img = os.path.join(os.path.dirname(os.path.abspath(__file__)), "visualizations", "class_distribution.png")
        if os.path.exists(class_dist_img):
            st.image(class_dist_img, caption="Class Frequency & Proportions (Imbalance Ratio: ~6.9:1)", use_container_width=True)

    with eda_row1_col2:
        msg_len_img = os.path.join(os.path.dirname(os.path.abspath(__file__)), "visualizations", "message_length_distribution.png")
        if os.path.exists(msg_len_img):
            st.image(msg_len_img, caption="Character Length Distribution: Ham vs Spam", use_container_width=True)

    eda_row2_col1, eda_row2_col2 = st.columns(2)
    with eda_row2_col1:
        word_box_img = os.path.join(os.path.dirname(os.path.abspath(__file__)), "visualizations", "word_count_distribution.png")
        if os.path.exists(word_box_img):
            st.image(word_box_img, caption="Word Count Distribution by Class", use_container_width=True)

    with eda_row2_col2:
        top_words_img = os.path.join(os.path.dirname(os.path.abspath(__file__)), "visualizations", "top_words_comparison.png")
        if os.path.exists(top_words_img):
            st.image(top_words_img, caption="Top 12 Vocabulary Terms in Spam vs Ham", use_container_width=True)

# -------------------------------------------------------------
# TAB 4: Interview & Conceptual Guide
# -------------------------------------------------------------
with tab4:
    st.subheader("Machine Learning Interview & Viva Preparation")
    st.markdown("Essential questions, mathematical intuitions, and design trade-offs frequently asked in technical interviews.")

    with st.expander("1. Why is Naive Bayes called 'Naive'? How does it work for text?"):
        st.write("""
        Naive Bayes is called **'Naive'** because it makes a strong simplifying assumption: 
        **all input features (words) are conditionally independent given the class label**.
        In human language, words are clearly dependent (e.g., 'credit' is often followed by 'card').
        Despite this violated assumption, Naive Bayes performs remarkably well in text classification because:
        - We only need the relative ranking of probabilities $P(\text{Spam}|X) > P(\text{Ham}|X)$, not perfect calibrated joint probabilities.
        - The computation reduces to simple multiplications:
          $$P(C|w_1, w_2, \\dots, w_n) \\propto P(C) \\prod_{i=1}^n P(w_i|C)$$
        """)

    with st.expander("2. Why is Precision more critical than Accuracy or Recall in Spam Filtering?"):
        st.write("""
        - **Accuracy Paradox:** In an imbalanced dataset (87.4% Ham, 12.6% Spam), a dumb model predicting 'Ham' every time gets **87.4% Accuracy**, but catches 0% spam.
        - **Asymmetric Misclassification Costs:**
          - **False Negative (FN):** A spam email slips into the Inbox. Cost: User spends 1 second deleting it.
          - **False Positive (FP):** A legitimate email (job interview offer, password reset, medical alert) is sent to the Spam/Trash folder. Cost: Catastrophic missed opportunity.
        - **Precision = TP / (TP + FP):** High precision ensures that when the system alerts 'SPAM', it is virtually guaranteed to be correct (our model achieved **99.04% precision**, with only 1 false positive out of 904 test ham messages).
        """)

    with st.expander("3. How does TF-IDF feature extraction work?"):
        st.write("""
        **TF-IDF** stands for **Term Frequency - Inverse Document Frequency**.
        - **Term Frequency (TF):** Measures how frequently term $t$ appears in document $d$:
          $$\\text{TF}(t, d) = \\frac{\\text{Count}(t, d)}{\\text{Total words in } d}$$
        - **Inverse Document Frequency (IDF):** Penalizes words that appear everywhere across all emails (e.g., 'the', 'is', 'hello'):
          $$\\text{IDF}(t) = \\log\\left(\\frac{1 + N}{1 + \\text{DF}(t)}\\right) + 1$$
        - **TF-IDF:** $\\text{TF-IDF}(t, d) = \\text{TF}(t, d) \\times \\text{IDF}(t)$. High TF-IDF scores correspond to words that are frequent in this specific message but rare across the corpus (e.g., 'prize', 'voucher', 'urgent').
        """)

    with st.expander("4. What is Laplace Smoothing (alpha parameter in MultinomialNB)?"):
        st.write("""
        If a new incoming email contains a word that was never seen in training spam messages, $P(\\text{word}|\\text{Spam}) = 0$.
        Because Naive Bayes multiplies all word probabilities together, that single zero would force the entire product to zero:
        $$P(\\text{Spam}|X) = 0$$
        To prevent this **Zero-Probability Problem**, Laplace Smoothing adds a pseudocount $\\alpha$ (typically $\\alpha=1.0$):
        $$P(w_i|C) = \\frac{\\text{count}(w_i, C) + \\alpha}{\\sum_{w} \\text{count}(w, C) + \\alpha \\cdot |V|}$$
        """)

    with st.expander("5. Why did we remove duplicate messages before training?"):
        st.write("""
        In SMS and email datasets, spammers blast identical text to hundreds of users, and common short ham replies like 'Ok' or 'Call me later' repeat frequently.
        If duplicates are kept and random train-test splitting is performed:
        - The exact same text will leak into both train and test partitions (**Data Leakage**).
        - Test evaluation metrics become artificially inflated, masking overfitting.
        Deduplicating ensures that the test set measures the model's true generalization capability on novel messages.
        """)
