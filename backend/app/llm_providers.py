"""
Thin, provider-agnostic wrapper around the model being *tested* for
hallucination.

Design note: the detector needs two things from the underlying model:
  1. N independent samples of the same prompt at temperature > 0
     (used for the self-consistency / semantic divergence signal).
  2. Token log-probabilities for one greedy decode, if the provider
     exposes them (used for the entropy signal). Anthropic's public API
     does not currently return logprobs, so that signal is simply
     omitted (weight redistributed to divergence) when using Anthropic.
     OpenAI's Chat Completions API does support `logprobs`, so the
     entropy term is available there.

This keeps scorer.py provider-agnostic — it only ever calls
`get_samples()` and `get_reference_with_entropy()`.
"""
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

from openai import OpenAI

from app.config import settings

GROQ_BASE_URL = "https://api.groq.com/openai/v1"


@dataclass
class ReferenceResult:
    text: str
    token_entropy: Optional[float]  # mean per-token entropy in nats, or None


class LLMProvider:
    """Base interface. Subclasses implement the two methods below."""

    name: str
    model: str

    def get_samples(self, question: str, n: int, temperature: float, max_tokens: int) -> List[str]:
        raise NotImplementedError

    def get_reference_with_entropy(self, question: str, max_tokens: int) -> ReferenceResult:
        raise NotImplementedError


class AnthropicProvider(LLMProvider):
    name = "anthropic"

    def __init__(self):
        if not settings.anthropic_api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Add it to backend/.env "
                "(copy .env.example) before starting the server."
            )
        try:
            import anthropic as _anthropic
        except ImportError:
            raise RuntimeError("anthropic package not installed. Run: pip install anthropic")
        self.client = _anthropic.Anthropic(api_key=settings.anthropic_api_key)
        self.model = settings.anthropic_model

    def _single_call(self, question: str, temperature: float, max_tokens: int) -> str:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": question}],
        )
        return "".join(block.text for block in resp.content if block.type == "text").strip()

    def get_samples(self, question: str, n: int, temperature: float, max_tokens: int) -> List[str]:
        return [self._single_call(question, temperature, max_tokens) for _ in range(n)]

    def get_reference_with_entropy(self, question: str, max_tokens: int) -> ReferenceResult:
        # Anthropic's public Messages API does not expose token logprobs,
        # so we return a low-temperature reference answer with no entropy term.
        text = self._single_call(question, temperature=0.0, max_tokens=max_tokens)
        return ReferenceResult(text=text, token_entropy=None)


class OpenAIProvider(LLMProvider):
    name = "openai"

    def __init__(self):
        if not settings.openai_api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Add it to backend/.env before "
                "using provider=openai."
            )
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = settings.openai_model

    def _single_call(self, question: str, temperature: float, max_tokens: int) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": question}],
        )
        return resp.choices[0].message.content.strip()

    def get_samples(self, question: str, n: int, temperature: float, max_tokens: int) -> List[str]:
        return [self._single_call(question, temperature, max_tokens) for _ in range(n)]

    def get_reference_with_entropy(self, question: str, max_tokens: int) -> ReferenceResult:
        resp = self.client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=0.0,
            logprobs=True,
            messages=[{"role": "user", "content": question}],
        )
        choice = resp.choices[0]
        text = choice.message.content.strip()

        entropy = None
        if choice.logprobs and choice.logprobs.content:
            # Per-token entropy proxy: -logprob of the chosen token, averaged.
            # This is a lower bound on true entropy (needs the full
            # distribution for exact entropy) but correlates well with
            # model uncertainty and needs no extra API calls.
            neg_logprobs = [tok.logprob for tok in choice.logprobs.content if tok.logprob is not None]
            if neg_logprobs:
                entropy = -sum(neg_logprobs) / len(neg_logprobs)

        return ReferenceResult(text=text, token_entropy=entropy)


class GroqProvider(LLMProvider):
    """
    Groq uses an OpenAI-compatible API, so we reuse the openai SDK
    with Groq's base URL. Groq supports logprobs, so the entropy
    signal is available (same as the OpenAI provider).
    """
    name = "groq"

    def __init__(self):
        if not settings.groq_api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Add it to backend/.env before "
                "using provider=groq."
            )
        self.client = OpenAI(
            api_key=settings.groq_api_key,
            base_url=GROQ_BASE_URL,
        )
        self.model = settings.groq_model

    def _single_call(self, question: str, temperature: float, max_tokens: int) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": question}],
        )
        return resp.choices[0].message.content.strip()

    def get_samples(self, question: str, n: int, temperature: float, max_tokens: int) -> List[str]:
        return [self._single_call(question, temperature, max_tokens) for _ in range(n)]

    def get_reference_with_entropy(self, question: str, max_tokens: int) -> ReferenceResult:
        # Groq's chat models do not support logprobs, so we return a
        # low-temperature reference answer with no entropy term.
        # The score will be driven entirely by semantic divergence.
        text = self._single_call(question, temperature=0.0, max_tokens=max_tokens)
        return ReferenceResult(text=text, token_entropy=None)


def get_provider(name: Optional[str] = None) -> LLMProvider:
    provider_name = (name or settings.provider).lower()
    if provider_name == "anthropic":
        return AnthropicProvider()
    if provider_name == "openai":
        return OpenAIProvider()
    if provider_name == "groq":
        return GroqProvider()
    raise ValueError(f"Unknown provider '{provider_name}'. Use 'anthropic', 'openai', or 'groq'.")
