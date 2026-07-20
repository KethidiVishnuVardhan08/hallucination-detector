"""Pydantic models defining the API's request/response contracts."""
from typing import List, Optional
from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    question: str = Field(..., min_length=3, description="The prompt/question to test.")
    n_samples: Optional[int] = Field(
        None, ge=3, le=15, description="Override default number of samples (3-15)."
    )
    temperature: Optional[float] = Field(
        None, ge=0.0, le=2.0, description="Override sampling temperature."
    )


class SampleOut(BaseModel):
    index: int
    text: str


class AnalyzeResponse(BaseModel):
    question: str
    samples: List[SampleOut]
    reference_answer: str  # the greedy / temperature=0 answer, shown as "the" answer
    pairwise_similarity: List[List[float]]
    mean_pairwise_similarity: float
    semantic_divergence: float  # 1 - mean_pairwise_similarity, in [0,1]
    entropy_score: Optional[float]  # None if provider doesn't expose logprobs
    risk_score: float  # final calibrated 0-1 score
    verdict: str  # "likely grounded" | "uncertain" | "likely hallucinated"
    explanation: str
    provider: str
    model: str


class HealthResponse(BaseModel):
    status: str
    provider: str
    model: str
    embedding_backend: str
