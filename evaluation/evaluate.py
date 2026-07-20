"""
Evaluates the hallucination detector against the TruthfulQA benchmark and
reports real precision / recall / AUROC numbers.

METHODOLOGY (read this before you quote numbers on your resume):

  1. For each question, we get the model's reference answer plus the
     self-consistency risk score from our own detector (`app.scorer`).
     The detector does NOT see TruthfulQA's reference answers — it is
     ground-truth-free at inference time, exactly as it would be in
     production.

  2. To EVALUATE the detector, we need to know whether the reference
     answer was actually truthful or not. We build this pseudo-label by
     embedding the reference answer and comparing its similarity to
     TruthfulQA's author-written `correct_answers` vs `incorrect_answers`
     for that question. If it's closer to the incorrect set, we label it
     "hallucinated"; otherwise "truthful". This mirrors the automatic
     judging approach used in the original TruthfulQA paper (which used
     a fine-tuned GPT judge) but is a simplified, transparent proxy —
     be upfront about this limitation in your writeup/README.

  3. We then treat this as a binary classification problem: does our
     risk_score predict the pseudo-label? We report precision, recall,
     F1 at the configured threshold, plus AUROC (threshold-independent).

Usage:
    python evaluation/download_truthfulqa.py     # once, to fetch data
    python evaluation/evaluate.py --n 60          # run on 60 questions

Runs N live API calls x (n_samples+1) each, so keep --n modest for cost.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app.config import settings
from app.embeddings import embed_and_similarity
from app.llm_providers import get_provider
from app.scorer import score_question

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "truthfulqa.json"
RESULTS_PATH = Path(__file__).resolve().parent / "results.csv"
REPORT_PATH = Path(__file__).resolve().parent / "report.png"


def pseudo_label(reference_answer: str, correct: list, incorrect: list) -> str:
    """
    Ground-truth-only-for-evaluation labeling: is the reference answer
    semantically closer to the author-written correct or incorrect
    answer set? Returns "truthful" or "hallucinated".
    """
    if not correct or not incorrect:
        return "unknown"

    texts = [reference_answer] + correct + incorrect
    sim = embed_and_similarity(texts)
    ref_row = sim[0]

    correct_scores = ref_row[1 : 1 + len(correct)]
    incorrect_scores = ref_row[1 + len(correct) :]

    max_correct = float(np.max(correct_scores))
    max_incorrect = float(np.max(incorrect_scores))

    return "truthful" if max_correct >= max_incorrect else "hallucinated"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=60, help="Number of questions to evaluate")
    parser.add_argument("--provider", type=str, default=None, help="Override provider (anthropic/openai)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not DATA_PATH.exists():
        print(f"Dataset not found at {DATA_PATH}. Run download_truthfulqa.py first.")
        sys.exit(1)

    with open(DATA_PATH) as f:
        all_questions = json.load(f)

    rng = np.random.default_rng(args.seed)
    sample = rng.choice(all_questions, size=min(args.n, len(all_questions)), replace=False)

    provider = get_provider(args.provider)
    print(f"Evaluating {len(sample)} questions with provider={provider.name} model={provider.model}")

    rows = []
    for i, item in enumerate(sample):
        question = item["question"]
        print(f"[{i+1}/{len(sample)}] {question[:70]}...")

        try:
            result = score_question(provider=provider, question=question)
        except Exception as e:
            print(f"  skipped (API error: {e})")
            continue

        label = pseudo_label(result.reference_answer, item["correct_answers"], item["incorrect_answers"])
        if label == "unknown":
            continue

        rows.append(
            {
                "question": question,
                "category": item["category"],
                "reference_answer": result.reference_answer,
                "risk_score": result.risk_score,
                "semantic_divergence": result.semantic_divergence,
                "entropy_score": result.entropy_score,
                "pseudo_label": label,
                "is_hallucination": 1 if label == "hallucinated" else 0,
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_PATH, index=False)
    print(f"\nSaved raw results to {RESULTS_PATH}")

    if df["is_hallucination"].nunique() < 2:
        print("Not enough label diversity to compute precision/recall (try a larger --n).")
        return

    _report_metrics(df)


def _report_metrics(df: pd.DataFrame):
    from sklearn.metrics import (
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
        roc_curve,
        precision_recall_curve,
    )
    import matplotlib.pyplot as plt

    y_true = df["is_hallucination"].values
    y_score = df["risk_score"].values
    y_pred = (y_score >= settings.risk_threshold).astype(int)

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auroc = roc_auc_score(y_true, y_score)

    print("\n=== Evaluation results ===")
    print(f"N evaluated:      {len(df)}")
    print(f"Base rate (hallucinated): {y_true.mean():.2%}")
    print(f"Threshold:         {settings.risk_threshold}")
    print(f"Precision:         {precision:.3f}")
    print(f"Recall:            {recall:.3f}")
    print(f"F1:                {f1:.3f}")
    print(f"AUROC:             {auroc:.3f}")

    fpr, tpr, _ = roc_curve(y_true, y_score)
    prec_curve, rec_curve, _ = precision_recall_curve(y_true, y_score)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].plot(fpr, tpr, color="#0E5C56", linewidth=2, label=f"AUROC={auroc:.3f}")
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="#999")
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].set_title("ROC curve")
    axes[0].legend()

    axes[1].plot(rec_curve, prec_curve, color="#B9812B", linewidth=2)
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall curve")

    fig.suptitle(f"Hallucination detector evaluation — n={len(df)}")
    fig.tight_layout()
    fig.savefig(REPORT_PATH, dpi=150)
    print(f"\nSaved chart to {REPORT_PATH}")


if __name__ == "__main__":
    main()
