# Signal/Divergence — Ground-Truth-Free Hallucination Risk Detector

A tool that estimates whether an LLM's answer is likely a hallucination
**without needing to know the correct answer in advance** — the same
constraint you'd face in a real production system, where you can't
manually fact-check every response before it reaches a user.

Built because every team shipping LLM features runs into the same wall:
you're terrified of hallucination, but you have no cheap way to catch
it at inference time. This project implements and evaluates one
concrete, measurable answer to that problem.

## How it works

The core idea (inspired by the **SelfCheckGPT** line of research,
Manakul et al. 2023) is simple but effective: **ask the model the same
question multiple times at non-zero temperature, and measure how much
the answers agree with each other.**

- If the model actually knows a fact, independent samples tend to
  converge on the same content, even when the wording varies.
- If the model is confabulating, there's no real fact anchoring the
  generation, so independent samples tend to diverge — different runs
  invent different, often mutually inconsistent, details.

Concretely, the pipeline is:

1. **Self-consistency sampling** — sample the same prompt N times (default 6) at temperature 0.9.
2. **Semantic divergence** — embed all N samples and compute the pairwise cosine similarity matrix. Low average similarity → high divergence → higher hallucination risk. Embeddings (not exact string match) so paraphrases are correctly recognized as agreeing.
3. **Token entropy (optional)** — if the provider exposes log-probabilities (OpenAI does; Anthropic's public API currently does not), the mean negative log-prob of a greedy decode is used as a second, complementary uncertainty signal.
4. **Combine into a single 0–1 risk score**, with a verdict (`likely grounded` / `uncertain` / `likely hallucinated`) and a plain-language explanation.

The full algorithm lives in [`backend/app/scorer.py`](backend/app/scorer.py) — it's deliberately decoupled from any specific provider or web framework so it can be unit-tested and reused by the evaluation script.

## Project structure

```
hallucination-detector/
├── backend/
│   ├── app/
│   │   ├── main.py            FastAPI app, serves API + frontend
│   │   ├── scorer.py           Core scoring algorithm
│   │   ├── llm_providers.py    Anthropic + OpenAI clients (pluggable)
│   │   ├── embeddings.py       sentence-transformers + TF-IDF fallback
│   │   ├── config.py           All tunable parameters
│   │   └── schemas.py          Pydantic request/response models
│   ├── tests/test_scorer.py    Unit tests (fake providers, no API key needed)
│   ├── requirements.txt
│   └── .env.example
├── frontend/
│   ├── index.html / style.css / app.js   Vanilla JS, no build step
├── evaluation/
│   ├── download_truthfulqa.py  Fetches the real TruthfulQA benchmark
│   └── evaluate.py             Runs the detector on it, reports precision/recall/AUROC
└── data/                       (created by download_truthfulqa.py)
```

## Setup

```bash
cd backend
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then add your ANTHROPIC_API_KEY (or OPENAI_API_KEY)
uvicorn app.main:app --reload --port 8000
```

Open **http://localhost:8000** — the FastAPI server serves the frontend directly, no separate frontend build step needed.

Try a question the model likely knows well (e.g. *"What is the capital of France?"*) and one it's likely to be shaky on (e.g. an obscure statistic, or a deliberately false-premise question like *"What year did Einstein win the Nobel Prize for relativity?"* — he won it for the photoelectric effect, not relativity, which is exactly the kind of subtle false-premise question that trips models up).

### Running the unit tests

```bash
cd backend
EMBEDDING_BACKEND=tfidf pytest tests/ -v
```

These use fake in-memory providers, so they run in ~2 seconds with no API key and no network access — useful to verify the algorithm logic itself is correct before you spend API budget on it.

## Running the benchmark evaluation (the part that gives you real numbers)

```bash
cd evaluation
python3 download_truthfulqa.py      # fetches the real TruthfulQA CSV from GitHub
python3 evaluate.py --n 60           # runs the detector on 60 questions
```

This produces:
- `evaluation/results.csv` — raw per-question scores and labels
- `evaluation/report.png` — ROC curve + precision-recall curve
- Console output with **precision, recall, F1, and AUROC**

**Be upfront about the methodology when you present this** (this is exactly the kind of thing an interviewer will probe, and being precise about it is a strength, not a weakness): to evaluate a ground-truth-free detector, you still need *some* ground truth for the test set. We use TruthfulQA's author-written `correct_answers` / `incorrect_answers` reference sets to build a pseudo-label for each generated response (closer to the correct set → "truthful", closer to incorrect → "hallucinated"). This is a simplified stand-in for the fine-tuned GPT-judge the original TruthfulQA paper used. The detector itself never sees these reference answers — they're used only to score the detector's output, exactly as a held-out test label would be used to evaluate any classifier.

## Honest limitations (know these before your interview)

- **Self-consistency catches confabulation, not confident falsehoods.** If a model is *consistently* wrong (e.g. a common misconception it always repeats the same way), this method won't flag it — divergence will be low because the model isn't confabulating, it's just wrong. This is a real, known limitation of the whole self-consistency family of methods, not a bug specific to this implementation.
- **N samples means N× the API cost and latency** per question. This makes it impractical to run on every production request as-is; a real deployment would likely use it selectively (e.g. only on high-stakes queries, or as an offline QA/audit tool rather than inline).
- **The entropy signal is provider-dependent.** Anthropic's public API doesn't expose logprobs, so with `provider=anthropic` the score is driven entirely by semantic divergence. This is documented in the code, not hidden.
- **The TruthfulQA pseudo-labeling is an approximation.** It's a legitimate, transparent evaluation methodology, but it's not the same as human-annotated ground truth — treat the reported numbers as a solid estimate, not a certified benchmark score.

## What I'd add next (good talking points, and genuinely worth building if you have more time)

1. **NLI-based contradiction detection** — instead of (or alongside) embedding cosine similarity, run a natural-language-inference model over each pair of samples to detect explicit contradiction rather than just topical drift. This is the more sophisticated variant used in the original SelfCheckGPT paper and typically improves precision.
2. **Retrieval cross-checking** — for factual questions, retrieve a supporting passage (Wikipedia API, or a local corpus) and check whether the model's answer is entailed by it. This adds a genuinely different, complementary signal beyond self-consistency.
3. **Per-sentence scoring** for long-form answers, rather than one score per whole response — hallucinations are often localized to a single sentence in an otherwise accurate paragraph.
4. **Caching + async batched sampling** so the N calls run concurrently instead of sequentially, cutting latency roughly N×.
5. **A calibration study** across multiple models/providers to see how the risk threshold should shift by model family.

## References

- Manakul, Liusie & Gales (2023), *SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models* — the core method this project implements a simplified version of.
- Lin, Hilton & Evans (2022), *TruthfulQA: Measuring How Models Mimic Human Falsehoods* — the benchmark used for evaluation.
