"""
FastAPI entrypoint.

Run with:
    uvicorn app.main:app --reload --port 8000

Then open http://localhost:8000 in a browser — this serves the frontend
directly, so there's no separate frontend server or build step needed.
"""
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.config import settings
from app.llm_providers import get_provider
from app.scorer import score_question
from app.schemas import AnalyzeRequest, AnalyzeResponse, SampleOut, HealthResponse

app = FastAPI(
    title="Hallucination Risk Scorer",
    description=(
        "Ground-truth-free hallucination detection via self-consistency "
        "sampling and semantic divergence, with optional token-entropy signal."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health", response_model=HealthResponse)
def health():
    model = (
        settings.anthropic_model if settings.provider == "anthropic"
        else settings.groq_model if settings.provider == "groq"
        else settings.openai_model
    )
    return HealthResponse(
        status="ok",
        provider=settings.provider,
        model=model,
        embedding_backend=settings.embedding_backend,
    )


@app.post("/api/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    try:
        provider = get_provider()
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    try:
        result = score_question(
            provider=provider,
            question=req.question,
            n_samples=req.n_samples,
            temperature=req.temperature,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM provider call failed: {e}")

    return AnalyzeResponse(
        question=req.question,
        samples=[SampleOut(index=i, text=t) for i, t in enumerate(result.samples)],
        reference_answer=result.reference_answer,
        pairwise_similarity=result.pairwise_similarity,
        mean_pairwise_similarity=result.mean_pairwise_similarity,
        semantic_divergence=result.semantic_divergence,
        entropy_score=result.entropy_score,
        risk_score=result.risk_score,
        verdict=result.verdict,
        explanation=result.explanation,
        provider=provider.name,
        model=provider.model,
    )


# --- Serve the frontend (static single-page app) ---
FRONTEND_DIR = Path(__file__).resolve().parent.parent.parent / "frontend"

if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    @app.get("/")
    def serve_index():
        return FileResponse(str(FRONTEND_DIR / "index.html"))
