"""
Reads the already-computed results.csv and sweeps every possible threshold
to find the one that maximises precision while keeping recall >= 0.60.
Prints a table and the recommended RISK_THRESHOLD to put in .env.
No API calls are made.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, f1_score

CSV = Path(__file__).resolve().parent / "results.csv"

if not CSV.exists():
    print("results.csv not found — run evaluate.py first.")
    raise SystemExit(1)

df = pd.read_csv(CSV)
y_true = df["is_hallucination"].values
y_score = df["risk_score"].values

print(f"Loaded {len(df)} rows  |  base-rate hallucinated = {y_true.mean():.1%}\n")
print(f"{'Threshold':>10} {'Precision':>10} {'Recall':>10} {'F1':>8} {'Flagged':>9}")
print("-" * 52)

best = None
thresholds = np.arange(0.10, 0.95, 0.01)
for t in thresholds:
    y_pred = (y_score >= t).astype(int)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    flagged = int(y_pred.sum())
    print(f"{t:>10.2f} {prec:>10.3f} {rec:>10.3f} {f1:>8.3f} {flagged:>9d}")
    # Track best threshold for 0.75-0.80 precision band with recall >= 0.60
    if 0.75 <= prec <= 0.85 and rec >= 0.60:
        if best is None or f1 > best["f1"]:
            best = dict(threshold=t, precision=prec, recall=rec, f1=f1)

print()
if best:
    print("=" * 52)
    print(f"RECOMMENDED  RISK_THRESHOLD = {best['threshold']:.2f}")
    print(f"  Precision = {best['precision']:.1%}   Recall = {best['recall']:.1%}   F1 = {best['f1']:.3f}")
else:
    # Fallback: find threshold nearest 0.77 precision regardless of recall
    results = []
    for t in thresholds:
        y_pred = (y_score >= t).astype(int)
        if y_pred.sum() == 0:
            continue
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec  = recall_score(y_true, y_pred, zero_division=0)
        f1   = f1_score(y_true, y_pred, zero_division=0)
        results.append(dict(threshold=t, precision=prec, recall=rec, f1=f1))
    if results:
        best = max(results, key=lambda r: r["f1"])
        print("=" * 52)
        print(f"BEST AVAILABLE  RISK_THRESHOLD = {best['threshold']:.2f}")
        print(f"  Precision = {best['precision']:.1%}   Recall = {best['recall']:.1%}   F1 = {best['f1']:.3f}")
