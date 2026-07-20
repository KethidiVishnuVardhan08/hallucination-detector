"""
Core hallucination-risk scoring algorithm.

Method (ground-truth-free, inference-time):

1. SELF-CONSISTENCY SAMPLING
   Ask the model the same question N times at temperature > 0.
   Intuition (from the SelfCheckGPT line of work, Manakul et al. 2023):
   if the model actually "knows" the answer, independent samples will
   converge on the same content even though wording varies. If the
   model is confabulating, samples tend to diverge because there is no
   real fact anchoring the generation — different samples invent
   different (mutually inconsistent) details.

2. SEMANTIC DIVERGENCE
   Embed all N samples and compute the pairwise cosine similarity
   matrix. Low average pairwise similarity => high divergence => higher
   hallucination risk. This is more robust than exact string match
   because it captures paraphrases ("Canberra" vs "the capital is
   Canberra, a planned city south-west of Sydney").

3. TOKEN ENTROPY (optional, provider-dependent)
   When the provider exposes log-probabilities for a greedy decode
   (currently: OpenAI), we also compute the mean negative log-prob of
   the chosen tokens as a proxy for the model's own uncertainty at
   generation time. This is a complementary signal: a model can be
   internally uncertain (high entropy) even on a question where it
   happens to sample consistently, and vice versa.

4. COMBINATION
   risk_score = w_div * semantic_divergence + w_ent * normalized_entropy
   Weights are renormalized when the entropy term is unavailable, so
   Anthropic-only runs still produce a valid 0-1 score driven entirely
   by the divergence signal.

This module is intentionally decoupled from any specific provider or
web framework — it operates on plain strings and numbers so it can be
unit-tested and reused by evaluation/evaluate.py without touching the
API layer.
"""
from dataclasses import dataclass, asdict
from typing import List, Optional

import numpy as np

from app.config import settings
from app.embeddings import embed_and_similarity, mean_off_diagonal
from app.llm_providers import LLMProvider, ReferenceResult


@dataclass
class ScoreResult:
    samples: List[str]
    reference_answer: str
    pairwise_similarity: List[List[float]]
    mean_pairwise_similarity: float
    semantic_divergence: float
    entropy_score: Optional[float]
    risk_score: float
    verdict: str
    explanation: str


# Entropy values (mean negative log-prob per token, in nats) empirically
# tend to range roughly 0 (very confident) to ~4+ (very uncertain) for
# typical chat models. We squash to [0,1] with this ceiling; values above
# it just saturate at 1.0. Re-tune ENTROPY_SATURATION if you evaluate a
# different model family and find the distribution shifted.
ENTROPY_SATURATION = 3.0


def _normalize_entropy(entropy_nats: float) -> float:
    return float(min(entropy_nats / ENTROPY_SATURATION, 1.0))


def _verdict_from_score(score: float) -> str:
    if score < settings.risk_threshold - 0.15:
        return "likely grounded"
    if score > settings.risk_threshold + 0.15:
        return "likely hallucinated"
    return "uncertain"


def _explain(divergence: float, entropy: Optional[float], verdict: str) -> str:
    parts = []
    if divergence < 0.15:
        parts.append(f"the model's {settings.n_samples} independent answers were highly consistent with each other (divergence={divergence:.2f})")
    elif divergence < 0.4:
        parts.append(f"the model's answers showed moderate disagreement across samples (divergence={divergence:.2f})")
    else:
        parts.append(f"the model's independent answers diverged substantially in content (divergence={divergence:.2f})")

    if entropy is not None:
        if entropy > 0.6:
            parts.append(f"and the model's own token-level confidence was low (normalized entropy={entropy:.2f})")
        else:
            parts.append(f"while the model's token-level confidence was relatively high (normalized entropy={entropy:.2f})")

    verdict_sentence = {
        "likely grounded": "This pattern is consistent with a fact the model reliably knows.",
        "uncertain": "This pattern is ambiguous — treat the answer with moderate caution and verify independently.",
        "likely hallucinated": "This pattern is consistent with confabulation: the model may be generating a plausible-sounding but unreliable answer.",
    }[verdict]

    return " ".join(parts).capitalize() + ". " + verdict_sentence


def score_question(
    provider: LLMProvider,
    question: str,
    n_samples: Optional[int] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> ScoreResult:
    n = n_samples or settings.n_samples
    temp = temperature if temperature is not None else settings.sampling_temperature
    max_tok = max_tokens or settings.max_tokens_per_sample

    # Step 1: self-consistency sampling
    samples = provider.get_samples(question, n=n, temperature=temp, max_tokens=max_tok)

    # Step 2: semantic divergence
    sim_matrix = embed_and_similarity(samples)
    mean_sim = mean_off_diagonal(sim_matrix)
    divergence = float(np.clip(1.0 - mean_sim, 0.0, 1.0))

    # Step 3: optional entropy signal from a separate greedy reference call
    reference: ReferenceResult = provider.get_reference_with_entropy(question, max_tokens=max_tok)
    entropy_norm = _normalize_entropy(reference.token_entropy) if reference.token_entropy is not None else None

    # Step 4: combine, renormalizing weights if entropy is unavailable
    if entropy_norm is not None:
        risk = settings.weight_divergence * divergence + settings.weight_entropy * entropy_norm
    else:
        risk = divergence  # divergence alone, weight renormalized to 1.0

    risk = float(np.clip(risk, 0.0, 1.0))
    verdict = _verdict_from_score(risk)
    explanation = _explain(divergence, entropy_norm, verdict)

    return ScoreResult(
        samples=samples,
        reference_answer=reference.text,
        pairwise_similarity=sim_matrix.round(4).tolist(),
        mean_pairwise_similarity=round(mean_sim, 4),
        semantic_divergence=round(divergence, 4),
        entropy_score=round(entropy_norm, 4) if entropy_norm is not None else None,
        risk_score=round(risk, 4),
        verdict=verdict,
        explanation=explanation,
    )
