"""
Unit tests for the core scoring algorithm. These use fake providers so
they run instantly with no API key and no network access — run them with:

    cd backend
    EMBEDDING_BACKEND=tfidf pytest tests/ -v

(TF-IDF backend is used in tests to avoid downloading the embedding
model; the algorithm logic itself is identical regardless of backend.)
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.scorer import score_question
from app.llm_providers import LLMProvider, ReferenceResult


class ConsistentProvider(LLMProvider):
    """Simulates a model that gives the same (correct) answer every time."""
    name = "fake"
    model = "fake-consistent"

    def get_samples(self, question, n, temperature, max_tokens):
        return ["Canberra is the capital city of Australia."] * n

    def get_reference_with_entropy(self, question, max_tokens):
        return ReferenceResult(text="Canberra is the capital city of Australia.", token_entropy=None)


class InconsistentProvider(LLMProvider):
    """Simulates a model confabulating a different answer each sample."""
    name = "fake"
    model = "fake-inconsistent"

    ANSWERS = [
        "The moon is made of cheese.",
        "The moon is a hologram projected by governments.",
        "The moon formed from debris after a giant impact with early Earth.",
        "The moon is actually a hollow artificial satellite.",
        "Scientists agree the moon is a captured asteroid.",
        "The moon was created by an ancient civilization.",
    ]

    def get_samples(self, question, n, temperature, max_tokens):
        return self.ANSWERS[:n]

    def get_reference_with_entropy(self, question, max_tokens):
        return ReferenceResult(text=self.ANSWERS[0], token_entropy=None)


class EntropyAwareProvider(LLMProvider):
    """Simulates a provider that also exposes a high entropy value."""
    name = "fake"
    model = "fake-entropy"

    def get_samples(self, question, n, temperature, max_tokens):
        return ["A plausible-sounding but shaky answer."] * n

    def get_reference_with_entropy(self, question, max_tokens):
        return ReferenceResult(text="A plausible-sounding but shaky answer.", token_entropy=2.8)


def test_consistent_answers_score_low_risk():
    result = score_question(ConsistentProvider(), "What is the capital of Australia?", n_samples=6)
    assert result.risk_score < 0.2
    assert result.verdict == "likely grounded"


def test_inconsistent_answers_score_high_risk():
    result = score_question(InconsistentProvider(), "What is the moon made of?", n_samples=6)
    assert result.risk_score > 0.6
    assert result.verdict == "likely hallucinated"


def test_consistent_scores_lower_than_inconsistent():
    consistent = score_question(ConsistentProvider(), "q", n_samples=6)
    inconsistent = score_question(InconsistentProvider(), "q", n_samples=6)
    assert consistent.risk_score < inconsistent.risk_score


def test_entropy_signal_is_incorporated_when_available():
    result = score_question(EntropyAwareProvider(), "q", n_samples=4)
    assert result.entropy_score is not None
    assert result.entropy_score > 0.5  # 2.8 nats / 3.0 saturation ≈ 0.93


def test_entropy_is_none_when_provider_does_not_expose_it():
    result = score_question(ConsistentProvider(), "q", n_samples=4)
    assert result.entropy_score is None


def test_pairwise_similarity_matrix_shape():
    n = 5
    result = score_question(ConsistentProvider(), "q", n_samples=n)
    assert len(result.pairwise_similarity) == n
    assert all(len(row) == n for row in result.pairwise_similarity)


def test_risk_score_is_bounded():
    result = score_question(InconsistentProvider(), "q", n_samples=6)
    assert 0.0 <= result.risk_score <= 1.0
