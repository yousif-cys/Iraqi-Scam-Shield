"""
main_api.py
-----------
FastAPI backend server for the Pre-Transaction Anti-Scam Intervention Agent.
Iraqi Mobile Wallets – ZainCash / QiCard.

Endpoints
─────────
POST /api/v1/transaction/pre-evaluate
    Runs Layer-1 (RiskEngine) deterministic checks, then conditionally
    calls Layer-2 (IraqiAICoach / Groq LLM) when risk is high enough.

GET  /health
    Liveness probe – returns server uptime and component status.

Run:
    uvicorn main_api:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from ai_coach import IraqiAICoach
from risk_engine import RiskEngine

# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("main_api")

# ──────────────────────────────────────────────────────────────────────────────
# Application-level singletons (initialised once at startup)
# ──────────────────────────────────────────────────────────────────────────────

_risk_engine: RiskEngine | None = None
_ai_coach: IraqiAICoach | None = None
_server_start: float = 0.0


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle manager."""
    global _risk_engine, _ai_coach, _server_start
    logger.info("=== Server startup – loading components ===")
    _server_start = time.time()
    _risk_engine = RiskEngine("dataset.json")
    _ai_coach = IraqiAICoach()
    logger.info("=== All components ready ===")
    yield
    logger.info("=== Server shutdown ===")


# ──────────────────────────────────────────────────────────────────────────────
# FastAPI application
# ──────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Iraqi Anti-Scam Intervention API",
    description=(
        "Pre-Transaction Anti-Scam Intervention Agent for ZainCash / QiCard. "
        "Combines deterministic risk rules with Groq LLM Baghdadi-dialect advice."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # Tighten to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────────────────────────────────────
# Request / Response schemas
# ──────────────────────────────────────────────────────────────────────────────

class PreEvaluateRequest(BaseModel):
    """
    Incoming payload for the pre-transaction evaluation endpoint.
    """

    user_id: str = Field(
        ...,
        min_length=1,
        max_length=64,
        description="Wallet user identifier (e.g. 'U01', 'usr_abc123')",
        examples=["U01"],
    )
    recipient_phone: str = Field(
        ...,
        description="Iraqi recipient phone number (07xx-xxxxxxx format)",
        examples=["07901234567"],
    )
    amount_iqd: float = Field(
        ...,
        gt=0,
        description="Transfer amount in Iraqi Dinars",
        examples=[50000.0],
    )
    is_new_recipient: bool = Field(
        ...,
        description="True if this is the first transfer to this phone number",
    )
    user_historical_avg: float = Field(
        ...,
        ge=0,
        description="User's average transfer amount (IQD) over last 30 days",
        examples=[200000.0],
    )
    recent_transfer_count_10m: int = Field(
        ...,
        ge=0,
        description=(
            "Number of transfers this user has made in the last 10 minutes "
            "(including the current pending one)"
        ),
        examples=[1],
    )
    note: Optional[str] = Field(
        default=None,
        max_length=500,
        description="Transfer memo / reason as entered by user",
        examples=["رسوم تسليم الجائزة"],
    )

    @field_validator("recipient_phone")
    @classmethod
    def validate_iraqi_phone(cls, v: str) -> str:
        cleaned = v.strip()
        if not (cleaned.startswith("07") and len(cleaned) == 11 and cleaned.isdigit()):
            raise ValueError(
                f"recipient_phone must be an 11-digit Iraqi number starting with 07. "
                f"Got: '{v}'"
            )
        return cleaned

    @field_validator("user_historical_avg")
    @classmethod
    def validate_avg(cls, v: float) -> float:
        if v < 0:
            raise ValueError("user_historical_avg must be >= 0")
        return v


class AICoachAdvice(BaseModel):
    """Structured Baghdadi-dialect advice from the Groq LLM."""

    headline: str = Field(..., description="Short warning headline in Baghdadi dialect")
    scammer_next_move: str = Field(
        ..., description="Predicted psychological tactic the scammer is using right now"
    )
    recommended_action: str = Field(
        ..., description="Single, clear safe action step in Baghdadi dialect"
    )
    is_fallback: bool = Field(
        ..., description="True when LLM was unavailable and offline safety-net was used"
    )


class PreEvaluateResponse(BaseModel):
    """
    Full evaluation result returned to the wallet frontend.
    """

    tx_id: str = Field(..., description="Auto-generated transaction trace ID")
    requires_intervention: bool = Field(
        ..., description="True when risk_score >= 0.40 – UI must block and warn"
    )
    risk_score: float = Field(
        ..., ge=0.0, le=1.0, description="Aggregate risk score (0.0 – 1.0)"
    )
    triggered_rules: list[str] = Field(
        ..., description="List of deterministic rule IDs that fired"
    )
    rule_details: dict[str, str] = Field(
        default_factory=dict,
        description="Human-readable explanation per triggered rule",
    )
    latency_ms: float = Field(
        ..., description="Total end-to-end processing latency in milliseconds"
    )
    ai_coach_advice: Optional[AICoachAdvice] = Field(
        default=None,
        description="Groq LLM Baghdadi advice (null when intervention not required)",
    )
    evaluated_at: str = Field(
        ..., description="ISO-8601 UTC timestamp of evaluation"
    )


class HealthResponse(BaseModel):
    """Liveness probe response."""

    status: str
    uptime_seconds: float
    components: dict[str, str]
    timestamp: str


# ──────────────────────────────────────────────────────────────────────────────
# Global exception handler
# ──────────────────────────────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "internal_server_error",
            "detail": str(exc),
            "path": str(request.url.path),
        },
    )


# ──────────────────────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────────────────────

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness / readiness probe",
    tags=["Infrastructure"],
)
async def health_check() -> HealthResponse:
    """
    Returns server uptime and per-component readiness.
    Used by load balancers, Kubernetes probes, and monitoring dashboards.
    """
    return HealthResponse(
        status="ok",
        uptime_seconds=round(time.time() - _server_start, 2),
        components={
            "risk_engine": "ready" if _risk_engine is not None else "unavailable",
            "ai_coach":    "ready" if _ai_coach is not None else "unavailable",
        },
        timestamp=datetime.now(tz=timezone.utc).isoformat(),
    )


@app.post(
    "/api/v1/transaction/pre-evaluate",
    response_model=PreEvaluateResponse,
    status_code=status.HTTP_200_OK,
    summary="Pre-transaction anti-scam evaluation",
    tags=["Anti-Scam"],
    responses={
        422: {"description": "Validation error – malformed request body"},
        500: {"description": "Internal server error"},
    },
)
async def pre_evaluate(body: PreEvaluateRequest) -> PreEvaluateResponse:
    """
    **Execution Flow**

    1. Generate a unique ``tx_id`` and record start time.
    2. **Layer 1 – RiskEngine**: Evaluate deterministic rules against the
       user's historical baseline and request-supplied context.
    3. **Layer 2 – IraqiAICoach (Groq LLM)**: If ``requires_intervention``
       is ``True``, call the real Groq API for Baghdadi-dialect advice.
    4. Measure ``latency_ms`` and return the unified response.
    """
    t_start = time.perf_counter()
    tx_id = f"TX-{uuid.uuid4().hex[:12].upper()}"
    ts_now = datetime.now(tz=timezone.utc)
    ts_iso = ts_now.strftime("%Y-%m-%dT%H:%M:%SZ")

    logger.info(
        "PRE-EVAL START | tx=%s user=%s amount=%.0f IQD recipient=%s",
        tx_id, body.user_id, body.amount_iqd, body.recipient_phone,
    )

    # ── Layer 1: Deterministic Risk Engine ────────────────────────────────────
    assert _risk_engine is not None, "RiskEngine not initialised"

    assessment = _risk_engine.evaluate(
        tx_id=tx_id,
        user_id=body.user_id,
        recipient_phone=body.recipient_phone,
        amount_iqd=body.amount_iqd,
        timestamp=ts_iso,
        is_new_recipient=body.is_new_recipient,
        historical_avg_override=body.user_historical_avg if body.user_historical_avg > 0 else None,
        recent_count_override=body.recent_transfer_count_10m,
    )

    logger.info(
        "LAYER-1 | tx=%s score=%.2f intervention=%s rules=%s",
        tx_id, assessment.risk_score,
        assessment.requires_intervention, assessment.triggered_rules,
    )

    # ── Layer 2: AI Coach (Groq LLM) ─────────────────────────────────────────
    ai_advice_payload: Optional[AICoachAdvice] = None

    if assessment.requires_intervention:
        assert _ai_coach is not None, "IraqiAICoach not initialised"

        logger.info("LAYER-2 | tx=%s triggering Groq LLM call", tx_id)

        coach_response = _ai_coach.generate_advice(
            amount_iqd=body.amount_iqd,
            recipient_phone=body.recipient_phone,
            note=body.note,
            triggered_rules=assessment.triggered_rules,
            risk_score=assessment.risk_score,
        )

        ai_advice_payload = AICoachAdvice(
            headline=coach_response.headline,
            scammer_next_move=coach_response.scammer_next_move,
            recommended_action=coach_response.recommended_action,
            is_fallback=coach_response.is_fallback,
        )

        logger.info(
            "LAYER-2 | tx=%s LLM responded | is_fallback=%s",
            tx_id, coach_response.is_fallback,
        )

    # ── Latency measurement ───────────────────────────────────────────────────
    latency_ms = round((time.perf_counter() - t_start) * 1000, 2)

    logger.info(
        "PRE-EVAL DONE | tx=%s latency=%.2fms intervention=%s",
        tx_id, latency_ms, assessment.requires_intervention,
    )

    return PreEvaluateResponse(
        tx_id=tx_id,
        requires_intervention=assessment.requires_intervention,
        risk_score=round(assessment.risk_score, 4),
        triggered_rules=assessment.triggered_rules,
        rule_details=assessment.rule_details,
        latency_ms=latency_ms,
        ai_coach_advice=ai_advice_payload,
        evaluated_at=ts_now.isoformat(),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Dev runner
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
