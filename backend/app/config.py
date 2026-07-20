"""
Central configuration for the hallucination detector.

All tunable parameters live here so the scoring method can be adjusted
and re-evaluated without touching the algorithm code itself.
"""
import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    # --- LLM provider ---
    # "anthropic", "openai", or "groq" — the model being *tested* for hallucination.
    provider: str = os.getenv("LLM_PROVIDER", "groq")
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    anthropic_model: str = os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-20241022")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_model: str = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

    # --- Self-consistency sampling ---
    # Number of independent samples drawn for the same prompt.
    # More samples -> more stable estimate, but linearly more API cost/latency.
    n_samples: int = int(os.getenv("N_SAMPLES", "6"))
    sampling_temperature: float = float(os.getenv("SAMPLING_TEMPERATURE", "0.9"))
    max_tokens_per_sample: int = int(os.getenv("MAX_TOKENS_PER_SAMPLE", "220"))

    # --- Semantic divergence ---
    # "sentence-transformers" (local embedding model, better quality) or
    # "tfidf" (no downloads required, works fully offline, lower quality).
    embedding_backend: str = os.getenv("EMBEDDING_BACKEND", "sentence-transformers")
    embedding_model_name: str = os.getenv(
        "EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
    )

    # --- Risk score calibration ---
    # Final risk score = w_divergence * divergence + w_entropy * norm_entropy
    # entropy term only contributes when the provider exposes token logprobs.
    weight_divergence: float = float(os.getenv("WEIGHT_DIVERGENCE", "0.75"))
    weight_entropy: float = float(os.getenv("WEIGHT_ENTROPY", "0.25"))

    # Decision threshold on the final 0-1 risk score used to label a response
    # "likely hallucinated" vs "likely grounded". Tuned in evaluation/evaluate.py.
    risk_threshold: float = float(os.getenv("RISK_THRESHOLD", "0.45"))

    cors_origins: list = field(default_factory=lambda: ["*"])


settings = Settings()
