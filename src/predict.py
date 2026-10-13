"""
predict.py
==========
Standalone CLI and interactive testing module for Spam Email / SMS inference.
Loads the trained production Pipeline and generates predictions with confidence scores
and spam trigger keyword highlights.

Usage:
  - Interactive test mode (type/paste any email):
      python src/predict.py
      python src/predict.py --interactive
  - Single text prediction:
      python src/predict.py --text "Claim your $500 Amazon gift card now!"
  - Classify an email saved in a text file:
      python src/predict.py --file path/to/email.txt
  - Run demo test suite:
      python src/predict.py --demo
"""

import os
import sys
import argparse
import joblib

# Ensure src directory is in sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocess import clean_text

MODELS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"
)
PIPELINE_PATH = os.path.join(MODELS_DIR, "spam_classifier_pipeline.joblib")

# Top indicative spam keywords to check for explanation
SPAM_TRIGGER_KEYWORDS = {
    "free", "win", "winner", "cash", "prize", "claim", "urgent", "call", "txt", "text",
    "award", "won", "reward", "guaranteed", "selected", "credit", "stop", "nokia",
    "customer", "service", "reply", "discount", "offer", "congratulations", "urgent",
    "voucher", "unclaimed", "lottery", "gift", "card", "exclusive", "expire", "suspended"
}


def load_pipeline(path: str = PIPELINE_PATH):
    """Loads saved production pipeline."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model pipeline not found at {path}. Run train.py first.")
    return joblib.load(path)


def predict_message(message: str, pipeline=None) -> dict:
    """
    Classifies a single message as Spam or Ham.

    Args:
        message: Raw input text.
        pipeline: Optional pre-loaded scikit-learn pipeline.

    Returns:
        dict containing prediction label, confidence, probabilities, and trigger keywords.
    """
    if pipeline is None:
        pipeline = load_pipeline()

    cleaned = clean_text(message)

    # Fallback to lowercased text if cleaned is blank (e.g. only symbols or rare stopwords)
    text_to_eval = cleaned if cleaned.strip() else message.lower()

    pred_label_idx = pipeline.predict([text_to_eval])[0]
    label_str = "Spam" if pred_label_idx == 1 else "Ham"

    if hasattr(pipeline, "predict_proba"):
        probs = pipeline.predict_proba([text_to_eval])[0]
        ham_prob = float(probs[0])
        spam_prob = float(probs[1])
        confidence = spam_prob if pred_label_idx == 1 else ham_prob
    else:
        ham_prob = 0.0 if pred_label_idx == 1 else 1.0
        spam_prob = 1.0 if pred_label_idx == 1 else 0.0
        confidence = 1.0

    # Extract matching spam trigger keywords
    tokens = set(cleaned.split())
    triggers_found = sorted(list(tokens.intersection(SPAM_TRIGGER_KEYWORDS)))

    return {
        "original_text": message,
        "cleaned_text": cleaned,
        "prediction": label_str,
        "is_spam": bool(pred_label_idx == 1),
        "confidence": round(confidence * 100, 2),
        "ham_probability": round(ham_prob * 100, 2),
        "spam_probability": round(spam_prob * 100, 2),
        "trigger_words": triggers_found,
    }


# Ensure stdout handles UTF-8 on Windows terminals
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass


def print_prediction_result(res: dict):
    """Pretty prints prediction results in a formatted box."""
    tag = "[SPAM MESSAGE]" if res["is_spam"] else "[LEGITIMATE HAM]"
    border = "=" * 65

    print("\n" + border)
    print(f"  PREDICTION RESULT: {tag}")
    print(border)
    print(f"• Input Text      : \"{res['original_text'].strip()[:180]}{'...' if len(res['original_text']) > 180 else ''}\"")
    print(f"• Classification  : {res['prediction']}")
    print(f"• Confidence Score: {res['confidence']:.2f}%")
    print(f"• Probabilities   : Ham: {res['ham_probability']:.2f}%  |  Spam: {res['spam_probability']:.2f}%")
    if res["trigger_words"]:
        print(f"• Spam Triggers   : {', '.join(res['trigger_words'])}")
    else:
        print(f"• Spam Triggers   : None detected")
    print(border + "\n")


def interactive_mode(pipeline=None):
    """
    Interactive command-line loop where the user can type or paste any
    email or SMS message and get instant predictions.
    """
    if pipeline is None:
        pipeline = load_pipeline()

    print("\n" + "=" * 65)
    print("      EMAIL & SMS SPAM DETECTOR - INTERACTIVE CONSOLE")
    print("=" * 65)
    print("Type or paste any message below to test the spam classifier.")
    print("Type 'quit', 'exit', or 'q' to return/exit.\n")

    count = 1
    while True:
        try:
            user_input = input(f"[{count}] Enter email/message to test:\n> ").strip()
            if not user_input:
                print("[!] Please enter non-empty text.\n")
                continue

            if user_input.lower() in ["quit", "exit", "q"]:
                print("\nExiting interactive testing mode. Goodbye!\n")
                break

            result = predict_message(user_input, pipeline)
            print_prediction_result(result)
            count += 1

        except (KeyboardInterrupt, EOFError):
            print("\n\nInteractive testing session ended.")
            break


def run_demo_tests(pipeline=None):
    """Runs a suite of realistic sample emails/messages to demonstrate inference."""
    if pipeline is None:
        pipeline = load_pipeline()

    sample_tests = [
        "Hey, are we still meeting for lunch at 12:30 PM today?",
        "URGENT! You have won a 1,000,000 cash prize! Call 09061701461 to claim your reward immediately. Free entry in weekly competition.",
        "Can you please review the attached slide deck before tomorrow morning's team meeting?",
        "Congratulations! You have been selected for a free holiday voucher. Text CLAIM to 88088 now.",
        "Hi Mom, I will be home by 7 PM for dinner. Love you!",
        "FINAL NOTICE: Your bank account will be suspended. Verify your credit card details immediately by clicking here.",
        "Your Amazon order #402-99124 has been shipped and is scheduled for delivery on Thursday."
    ]

    print("=" * 70)
    print("           REAL-TIME PREDICTION TEST SUITE")
    print("=" * 70)

    for i, sample in enumerate(sample_tests, 1):
        res = predict_message(sample, pipeline)
        status_tag = "[SPAM]" if res["is_spam"] else "[HAM]"
        print(f"\nTest {i}: {status_tag} (Confidence: {res['confidence']:.2f}%)")
        print(f"Message  : \"{sample}\"")
        print(f"Probs    : Ham: {res['ham_probability']}% | Spam: {res['spam_probability']}%")
        if res["trigger_words"]:
            print(f"Triggers : {', '.join(res['trigger_words'])}")
    print("\n" + "=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Classify an Email or SMS message as Spam or Ham")
    parser.add_argument("--text", type=str, help="Email or message text to classify directly")
    parser.add_argument("--file", type=str, help="Path to a text file containing an email to classify")
    parser.add_argument("--demo", action="store_true", help="Run built-in sample demo predictions")
    parser.add_argument("--interactive", "-i", action="store_true", help="Launch interactive user input prompt")
    args = parser.parse_args()

    pipeline = load_pipeline()

    # Case 1: Specific text provided via CLI flag
    if args.text:
        res = predict_message(args.text, pipeline)
        print_prediction_result(res)
        return

    # Case 2: File path provided
    if args.file:
        if not os.path.exists(args.file):
            print(f"Error: File not found at '{args.file}'", file=sys.stderr)
            sys.exit(1)
        with open(args.file, "r", encoding="utf-8", errors="replace") as f:
            file_content = f.read()
        res = predict_message(file_content, pipeline)
        print_prediction_result(res)
        return

    # Case 3: Explicit demo flag
    if args.demo:
        run_demo_tests(pipeline)
        return

    # Case 4: Explicit interactive flag
    if args.interactive:
        interactive_mode(pipeline)
        return

    # Case 5: Default (no arguments passed) -> Provide user-friendly interactive menu
    print("=" * 65)
    print("          SPAM EMAIL CLASSIFIER - TESTING CONSOLE")
    print("=" * 65)
    print("Choose an option:")
    print("  [1] Interactive Mode: Test custom emails/messages (Type or Paste)")
    print("  [2] Demo Mode       : Run automated sample test suite")
    print("  [3] Exit")
    print("=" * 65)

    try:
        choice = input("Enter choice (1, 2, or 3) [default: 1]: ").strip()
        if choice in ["", "1"]:
            interactive_mode(pipeline)
        elif choice == "2":
            run_demo_tests(pipeline)
        elif choice == "3":
            print("Exiting. Have a great day!")
        else:
            print(f"Unknown choice '{choice}'. Launching interactive testing mode...")
            interactive_mode(pipeline)
    except (KeyboardInterrupt, EOFError):
        print("\nSession ended.")


if __name__ == "__main__":
    main()
