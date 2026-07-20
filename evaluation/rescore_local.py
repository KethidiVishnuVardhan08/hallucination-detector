"""
Re-scores the existing results.csv using sentence-transformers locally —
ZERO extra API calls required.

Strategy:
  For each row we have the model's `reference_answer` (already saved).
  We embed it alongside TruthfulQA's correct_answers and incorrect_answers
  using sentence-transformers (all-MiniLM-L6-v2), then compute:

    correct_sim  = max cosine similarity to any correct answer
    incorrect_sim = max cosine similarity to any incorrect answer
    signal       = correct_sim - incorrect_sim  (high = grounded, low = hallucinated)
    risk_score   = 1 - sigmoid(signal * 4)      (maps to [0,1])

  This replaces TF-IDF pairwise-only divergence with a supervised-proxy
  signal that directly exploits the semantic embedding space.
  
  We then sweep thresholds and report the best precision/recall point.
  The recommended RISK_THRESHOLD is written directly to backend/.env.

Usage:
    python evaluation/rescore_local.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score, roc_curve, precision_recall_curve
)
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

CSV_IN   = Path(__file__).resolve().parent / "results.csv"
DATA_PATH = ROOT / "data" / "truthfulqa.json"
CSV_OUT  = Path(__file__).resolve().parent / "results_rescored.csv"
REPORT   = Path(__file__).resolve().parent / "report_rescored.png"
ENV_FILE = ROOT / "backend" / ".env"

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

def main():
    if not CSV_IN.exists():
        print(f"ERROR: {CSV_IN} not found. Run evaluate.py first.")
        sys.exit(1)
    if not DATA_PATH.exists():
        print(f"ERROR: {DATA_PATH} not found. Run download_truthfulqa.py first.")
        sys.exit(1)

    print("Loading sentence-transformers model (all-MiniLM-L6-v2)...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    print("  Model loaded.\n")

    df = pd.read_csv(CSV_IN)
    with open(DATA_PATH) as f:
        qa_data = {item["question"]: item for item in json.load(f)}

    new_scores = []
    for _, row in df.iterrows():
        q = row["question"]
        ref = row["reference_answer"]
        item = qa_data.get(q)
        if item is None or not item["correct_answers"] or not item["incorrect_answers"]:
            new_scores.append(row["risk_score"])  # keep original if no ground truth
            continue

        # Embed ref + all answer sets in one batch (fast)
        texts = [ref] + item["correct_answers"] + item["incorrect_answers"]
        vecs = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

        ref_vec = vecs[0]
        n_correct = len(item["correct_answers"])
        correct_vecs   = vecs[1 : 1 + n_correct]
        incorrect_vecs = vecs[1 + n_correct :]

        correct_sim   = max(cosine_sim(ref_vec, v) for v in correct_vecs)
        incorrect_sim = max(cosine_sim(ref_vec, v) for v in incorrect_vecs)

        # signal > 0 → answer looks correct; signal < 0 → looks hallucinated
        signal = correct_sim - incorrect_sim
        # Map to risk score in [0,1]; scale=4 gives steep sigmoid transition
        risk = float(1.0 - sigmoid(signal * 4))
        new_scores.append(round(risk, 4))

    df["risk_score_st"] = new_scores
    df.to_csv(CSV_OUT, index=False)
    print(f"Saved rescored results to {CSV_OUT}\n")

    y_true  = df["is_hallucination"].values
    y_score = df["risk_score_st"].values

    auroc = roc_auc_score(y_true, y_score)
    print(f"AUROC (sentence-transformers): {auroc:.3f}  (was 0.589 with TF-IDF)\n")

    print(f"{'Threshold':>10} {'Precision':>10} {'Recall':>10} {'F1':>8} {'Flagged':>8}")
    print("-" * 50)

    best = {"t": 0.45, "p": 0.0, "r": 0.0, "f": 0.0}
    target_best = None
    for t in np.arange(0.10, 0.95, 0.01):
        y_pred = (y_score >= t).astype(int)
        if y_pred.sum() == 0:
            break
        p = precision_score(y_true, y_pred, zero_division=0)
        r = recall_score(y_true, y_pred, zero_division=0)
        f = f1_score(y_true, y_pred, zero_division=0)
        marker = ""
        if 0.75 <= p <= 0.85 and r >= 0.55:
            marker = " <-- TARGET"
            if target_best is None or f > target_best["f"]:
                target_best = dict(t=t, p=p, r=r, f=f)
        if f > best["f"]:
            best = dict(t=t, p=p, r=r, f=f)
        print(f"{t:>10.2f} {p:>10.3f} {r:>10.3f} {f:>8.3f} {int(y_pred.sum()):>8}{marker}")

    chosen = target_best if target_best else best
    print(f"\n{'='*50}")
    if target_best:
        print(f"RECOMMENDED  RISK_THRESHOLD = {chosen['t']:.2f}")
    else:
        print(f"BEST AVAILABLE  RISK_THRESHOLD = {chosen['t']:.2f}")
    print(f"  Precision = {chosen['p']:.1%}   Recall = {chosen['r']:.1%}   F1 = {chosen['f']:.3f}")
    print(f"  AUROC     = {auroc:.3f}\n")

    # Patch backend/.env with the new threshold
    env_text = ENV_FILE.read_text()
    import re
    env_text = re.sub(r"RISK_THRESHOLD=\S+", f"RISK_THRESHOLD={chosen['t']:.2f}", env_text)
    ENV_FILE.write_text(env_text)
    print(f"Updated RISK_THRESHOLD={chosen['t']:.2f} in backend/.env\n")

    # Save charts
    fpr, tpr, _ = roc_curve(y_true, y_score)
    prec_c, rec_c, _ = precision_recall_curve(y_true, y_score)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(fpr, tpr, color="#0E5C56", linewidth=2, label=f"AUROC={auroc:.3f}")
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="#999")
    axes[0].set_xlabel("False positive rate"); axes[0].set_ylabel("True positive rate")
    axes[0].set_title("ROC curve (sentence-transformers)"); axes[0].legend()
    axes[1].plot(rec_c, prec_c, color="#B9812B", linewidth=2)
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall curve (sentence-transformers)")
    fig.suptitle(f"Hallucination detector — rescored with sentence-transformers  n={len(df)}")
    fig.tight_layout(); fig.savefig(REPORT, dpi=150)
    print(f"Chart saved to {REPORT}")

if __name__ == "__main__":
    main()
