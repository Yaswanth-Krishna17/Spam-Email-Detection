"""
test_mail.py
============
Interactive email and SMS spam testing tool.
Allows users to type, paste, or pass any email or message to classify in real-time.

Usage:
  - Interactive test mode (type/paste any email):
      python test_mail.py

  - Pass message directly as argument:
      python test_mail.py "Congratulations! You won a $1,000 cash prize."
      python test_mail.py "Hi Team, please find attached the weekly project update."

  - Classify an email saved in a text file:
      python test_mail.py --file email.txt

  - Run automated demo suite:
      python test_mail.py --demo
"""

import sys
import os

# Ensure UTF-8 output on Windows terminals
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

# Add src to system path
SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from predict import predict_message, print_prediction_result, interactive_mode, run_demo_tests, load_pipeline


def main():
    pipeline = load_pipeline()

    # Check CLI arguments
    args = sys.argv[1:]

    if not args:
        # Default: Launch interactive typing prompt
        interactive_mode(pipeline)
        return

    if args[0] in ["--demo", "-d"]:
        run_demo_tests(pipeline)
        return

    if args[0] in ["--file", "-f"]:
        if len(args) < 2:
            print("Error: Please specify the file path, e.g. python test_mail.py --file email.txt")
            return
        filepath = args[1]
        if not os.path.exists(filepath):
            print(f"Error: File not found at '{filepath}'")
            return
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        res = predict_message(content, pipeline)
        print_prediction_result(res)
        return

    if args[0] in ["--interactive", "-i"]:
        interactive_mode(pipeline)
        return

    # User passed a message directly: python test_mail.py "hello world"
    input_text = " ".join(args)
    result = predict_message(input_text, pipeline)
    print_prediction_result(result)


if __name__ == "__main__":
    main()
