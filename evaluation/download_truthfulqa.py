"""
Downloads the real TruthfulQA benchmark (Lin, Hilton & Evans, 2022)
directly from the official GitHub repository.

TruthfulQA gives, for each question, a set of `correct_answers` and
`incorrect_answers` written by the dataset authors. We use these
reference sets ONLY to construct ground-truth labels for evaluating our
detector's precision/recall — the detector itself never sees them at
inference time, which is the whole point of a ground-truth-free method.

Usage:
    python evaluation/download_truthfulqa.py
"""
import csv
import io
import json
import urllib.request
from pathlib import Path

URL = "https://raw.githubusercontent.com/sylinrl/TruthfulQA/main/TruthfulQA.csv"
OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "truthfulqa.json"


def main():
    print(f"Downloading TruthfulQA from {URL} ...")
    with urllib.request.urlopen(URL) as resp:
        raw = resp.read().decode("utf-8")

    reader = csv.DictReader(io.StringIO(raw))
    rows = []
    for row in reader:
        rows.append(
            {
                "category": row.get("Category", ""),
                "question": row["Question"],
                "best_answer": row.get("Best Answer", ""),
                "correct_answers": [
                    a.strip() for a in row.get("Correct Answers", "").split(";") if a.strip()
                ],
                "incorrect_answers": [
                    a.strip() for a in row.get("Incorrect Answers", "").split(";") if a.strip()
                ],
            }
        )

    OUT_PATH.parent.mkdir(exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(rows, f, indent=2)

    print(f"Saved {len(rows)} questions to {OUT_PATH}")


if __name__ == "__main__":
    main()
