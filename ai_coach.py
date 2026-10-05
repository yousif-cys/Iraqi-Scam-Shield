"""
ai_coach.py
-----------
IraqiAICoach – Pre-Transaction Anti-Scam Intervention Agent
for Iraqi Mobile Wallets (ZainCash / QiCard).

Sends a real-time prompt to Groq (llama-3.3-70b-versatile) and returns a
structured JSON warning in authentic Baghdadi Iraqi Arabic dialect.
Uses Pydantic for strict schema verification and output safety.

Decision Authority Rules (NON-NEGOTIABLE):
  - Layer 1 (RiskEngine) owns `requires_intervention`. Layer 2 (this module)
    NEVER modifies, re-evaluates, or overrides that flag.
  - Layer 2 is only invoked when requires_intervention is already True.
  - All Groq outputs are validated via Pydantic before being returned.
  - Any validation or API failure falls back to a pre-verified safe response.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, List, Optional

# pyrefly: ignore [missing-import]
from pydantic import BaseModel, Field, ValidationError, field_validator
# pyrefly: ignore [missing-import]
from groq import APIConnectionError, APIStatusError, APITimeoutError

from groq_config import groq_client, GROQ_MODEL, GROQ_TEMPERATURE, GROQ_MAX_TOKENS

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Pydantic Output Safety Schema
# ──────────────────────────────────────────────────────────────────────────────

class IraqiCoachAdviceSchema(BaseModel):
    """
    Enforces 100% JSON Schema compliance for every Groq LLM output.
    Any field violation raises ValidationError → safe fallback is returned.
    """

    headline: str = Field(
        ...,
        min_length=1,
        description="Short warning headline in authentic Baghdadi dialect.",
    )
    scammer_next_move: str = Field(
        ...,
        min_length=1,
        description="Prediction of the psychological manipulation tactic the scammer is deploying.",
    )
    what_looks_wrong:str = Field(
        ...,
        min_length=1,
        description="Explanation of what seems suspicious about the transaction.",
    )
    recommended_action: str = Field(
        ...,
        min_length=1,
        description="Single, crystal-clear safe action step in Baghdadi dialect.",
    )
    confidence_level: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Model confidence score [0.0 – 1.0].",
    )

    @field_validator("headline", "scammer_next_move", "recommended_action", "what_looks_wrong")
    @classmethod
    def validate_non_empty(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Field cannot be empty or whitespace-only.")
        return cleaned


# ──────────────────────────────────────────────────────────────────────────────
# Validated response returned to callers
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ScamInterventionResponse:
    """
    Immutable, fully validated response returned to the wallet UI or benchmark.

    Attributes
    ----------
    headline:
        Short warning headline in Baghdadi dialect.
    what_looks_wrong:
        explanation of what seems suspicious about the transaction.    
    scammer_next_move:
        Prediction of the psychological manipulation tactic in play.
    recommended_action:
        A single, crystal-clear safe action step for the user.
    confidence_level:
        Confidence score in [0.0, 1.0].
    is_fallback:
        True when the API was unreachable or schema validation failed.
    raw_json:
        Original JSON string returned by the LLM (empty string on fallback).
    """

    headline: str
    scammer_next_move: str
    recommended_action: str
    what_looks_wrong: str 
    confidence_level: float = 0.9
    is_fallback: bool = False
    raw_json: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "headline": self.headline,
            "what_looks_wrong": self.what_looks_wrong,
            "scammer_next_move": self.scammer_next_move,
            "recommended_action": self.recommended_action,
            "confidence_level": self.confidence_level,
            "is_fallback": self.is_fallback,
        }


# ──────────────────────────────────────────────────────────────────────────────
# System prompt (Baghdadi dialect enforced)
# ──────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are the security assistant inside the Zain Cash app.
Your job is to speak to the user in natural Iraqi Arabic like a person who genuinely cares about their safety — not like a bank, police officer, or formal security system.
A transaction has been paused because it shows signs that may be consistent with a scam. Your role is to briefly explain what looks unusual, why it matters, what the scammer may say or do next, and one safe next step.
The input you receive may include:
* Transaction amount
* Recipient
* Triggered detection rules
* Suspected scam type, or "unspecified"
CORE RULES:
1. LANGUAGE
* Output must be "natural Iraqi Arabic".
* Use short, clear sentences.
* Avoid formal banking, legal, or technical language.
* The term "OTP" is allowed.
* Do not use Modern Standard Arabic unless necessary for clarity.
2. USE ONLY PROVIDED INFORMATION
* Base your response only on the information provided in the input.
* Never invent names, amounts, events, relationships, previous conversations, or other details.
* Do not assume facts that are not explicitly provided.
3. SCAM TYPE
* If a specific scam type is provided, describe the likely scammer behavior based on that type.
* If the scam type is "unspecified", do not invent a specific scenario.
* Instead, mention general warning signs such as pressure, urgency, secrecy, or unusual payment requests when they are relevant to the provided information.
4. AVOID FALSE CERTAINTY
* Do not state that the user is definitely being scammed.
* Prefer wording such as:
  "هذا يشبه محاولة احتيال"
  "التحويل بيه إشارات مريبة"
  "هذا السلوك ممكن يكون مرتبط باحتيال"
* Only describe something as highly suspicious when the provided rules strongly support that conclusion.
5. TONE
* Sound caring, calm, and protective.
* Do not sound threatening, judgmental, robotic, or overly formal.
* Do not command, pressure, shame, or scare the user.
* Explain the risk and let the user make the final decision.
6. PRIVACY AND SECURITY
* Never ask the user for OTPs, passwords, PINs, card numbers, or other sensitive personal information.
* Never request unnecessary personal details.
7. VARIATION
* Avoid repeating the exact same wording across responses.
* Vary sentence structure and phrasing naturally while keeping the meaning consistent.
8. FIELD LENGTH
* Each field should contain one or two short sentences.
* Keep the overall response concise and easy to read on a mobile screen.
9. RECOMMENDED ACTION
* Provide exactly one practical and safe next step.
* Phrase it as advice, not a command.
* Example style:
  "الأفضل تتأكد من الشخص من وسيلة ثانية قبل تكمل."
* Do not provide multiple actions in this field.
10. SCAMMER NEXT MOVE
* This field is a prediction, not a fact.
* Describe what the scammer may say or try next.
* Use wording such as:
  "ممكن بعدها يكول..."
  "غالبًا يحاول يقنعك..."
  "قد يضغط عليك بـ..."
OUTPUT FORMAT:
Return valid JSON only.
Do not include markdown, code fences, explanations, or any text outside the JSON.
The JSON must contain exactly these four keys:
{
"headline": "عنوان تنبيه قصير",
"what_looks_wrong": "شنو الغلط بهالتحويل بالتحديد، وليش، من المعلومات المعطاة فقط",
"scammer_next_move": "شنو ممكن يحچي أو يسوي المحتال بعدين، حسب نوع الاحتيال أو الإشارات المعطاة",
"recommended_action": "خطوة واحدة واضحة وآمنة بصيغة نصيحة",
"confidence_level": 0.0-1.0
}
IMPORTANT:
* Keep the JSON valid.
* Use double quotes for all JSON keys and string values.
* Do not add extra keys.
* Do not write any text outside the JSON.
""".strip()


# ──────────────────────────────────────────────────────────────────────────────
# Main agent class
# ──────────────────────────────────────────────────────────────────────────────

class IraqiAICoach:
    """
    Layer 2 – Pre-Transaction Anti-Scam Intervention Agent.

    Strict contract:
    ─────────────────────────────────────────────────────────
    • Uses Groq SDK (`groq.Client`) directly.
    • Forces `response_format={"type": "json_object"}` on every call.
    • Validates every LLM response through `IraqiCoachAdviceSchema` (Pydantic).
    • Any error (API, timeout, parse, schema) triggers `_fallback_response()`.
    • NEVER reads, modifies, or re-computes `requires_intervention`.
    """

    def __init__(
        self,
        model: str = GROQ_MODEL,
        temperature: float = GROQ_TEMPERATURE,
        max_tokens: int = GROQ_MAX_TOKENS,
    ) -> None:
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        logger.info(
            "IraqiAICoach ready | model=%s | temp=%.2f | max_tokens=%d",
            self._model,
            self._temperature,
            self._max_tokens,
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def get_coach_advice(
        self,
        assessment: Any,
        amount_iqd: float,
        recipient_phone: str,
        note: Optional[str] = None,
    ) -> ScamInterventionResponse:
        """
        Routes through Layer 1 decision gate before calling LLM.

        DECISION AUTHORITY ISOLATION:
            Layer 2 reads `assessment.requires_intervention` but NEVER alters it.
            If requires_intervention is False → return fallback immediately.
            LLM is only invoked when Layer 1 has already decided True.
        """
        requires_intervention = getattr(assessment, "requires_intervention", False)
        if not requires_intervention:
            return self._fallback_response("no_intervention_required")

        triggered = getattr(
            assessment, "triggered_rules", getattr(assessment, "triggers", [])
        )
        score = getattr(assessment, "risk_score", 0.0)

        return self.generate_advice(
            amount_iqd=amount_iqd,
            recipient_phone=recipient_phone,
            note=note,
            triggered_rules=triggered,
            risk_score=score,
        )

    def generate_advice(
        self,
        *,
        amount_iqd: float,
        recipient_phone: str,
        note: Optional[str] = None,
        triggered_rules: Optional[List[str]] = None,
        risk_score: float = 0.0,
    ) -> ScamInterventionResponse:
        """
        Primary entrypoint called by the wallet pipeline when
        ``requires_intervention == True``.

        Builds a context-rich Arabic prompt from Layer 1 risk signals and
        returns a Pydantic-validated `ScamInterventionResponse`.
        """
        rule_labels: dict[str, str] = {
            "RULE_01_NEW_RECIPIENT": "تحويل كبير لمستلم جديد",
            "RULE_02_HIGH_VELOCITY":           "سرعة تحويلات غير طبيعية لنفس الشخص",
            "RULE_03_ABNORMAL_AMOUNT" :        "مبلغ اكبر من المعتاد",
            "RULE_04_KNOWN_SCAM_FEE":          "مبلغ رسوم نصب معروف", 
            "RULE_05_SUSPICIOUS_TIME" :               "وقت غير معتاد"
        }
        risk_signals: List[str] = [
            rule_labels.get(r, r) for r in (triggered_rules or [])
        ]
        risk_signals.append(f"نسبة الخطر: {risk_score:.0%}")

        return self.evaluate_transaction(
            amount_iqd=amount_iqd,
            recipient_name=recipient_phone,
            recipient_account=recipient_phone,
            transaction_note=note,
            risk_signals=risk_signals,
        )

    def evaluate_transaction(
        self,
        *,
        amount_iqd: float,
        recipient_name: str,
        recipient_account: str,
        transaction_note: Optional[str] = None,
        risk_signals: Optional[List[str]] = None,
    ) -> ScamInterventionResponse:
        """
        Builds the user prompt, calls Groq SDK with `json_object` response format,
        validates the output through Pydantic, and returns a typed response.
        Falls back gracefully on any error.
        """
        user_prompt = self._build_user_prompt(
            amount_iqd=amount_iqd,
            recipient_name=recipient_name,
            recipient_account=recipient_account,
            transaction_note=transaction_note,
            risk_signals=risk_signals or [],
        )

        logger.info(
            "Sending transaction evaluation | amount=%.0f IQD | recipient=%s",
            amount_iqd,
            recipient_name,
        )

        try:
            return self._call_groq(user_prompt)

        except APITimeoutError as exc:
            logger.error("Groq request timed out: %s", exc)
            return self._fallback_response("timeout")

        except APIConnectionError as exc:
            logger.error("Groq connection error: %s", exc)
            return self._fallback_response("connection")

        except APIStatusError as exc:
            logger.error("Groq API status error %d: %s", exc.status_code, exc.message)
            return self._fallback_response("api_error")

        except ValidationError as exc:
            logger.error("Pydantic schema validation failed: %s", exc)
            return self._fallback_response("schema_validation_error")

        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            logger.error("Failed to parse LLM JSON response: %s", exc)
            return self._fallback_response("parse_error")

        except Exception as exc:  # noqa: BLE001
            logger.exception("Unexpected error during Groq call: %s", exc)
            return self._fallback_response("unknown")

    # ── Private helpers ────────────────────────────────────────────────────────

    def _build_user_prompt(
        self,
        *,
        amount_iqd: float,
        recipient_name: str,
        recipient_account: str,
        transaction_note: Optional[str],
        risk_signals: List[str],
    ) -> str:
        """Constructs the Arabic-language prompt with Layer 1 risk signal context."""
        signals_text = "، ".join(risk_signals) if risk_signals else "مو محدد"
        note_text = transaction_note or "ما كو ملاحظة"

        return (
            f"المستخدم يريد يحول:\n"
            f"- المبلغ: {amount_iqd:,.0f} دينار عراقي\n"
            f"- المستلم: {recipient_name}\n"
            f"- رقم الحساب: {recipient_account}\n"
            f"- سبب التحويل: {note_text}\n"
            f"- إشارات خطر مكتشفة: {signals_text}\n\n"
            f"حلّل هاي المعاملة وحذّر المستخدم باللهجة العراقية "
            f"اتنبأ بالخطوة النفسية الجاية للنصاب بالهاتف وعطِ توصية واحدة واضحة."
        )

    def _call_groq(self, user_prompt: str) -> ScamInterventionResponse:
        """
        Executes the Groq SDK call with `response_format={"type": "json_object"}`.
        Validates the returned JSON through `IraqiCoachAdviceSchema` (Pydantic).
        Raises on any validation or API error — callers handle fallback.
        """
        completion = groq_client.chat.completions.create(
            model=self._model,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
        )

        raw_content: str = completion.choices[0].message.content or ""
        logger.debug("Raw LLM response: %s", raw_content)

        # Strict Pydantic validation — raises ValidationError on schema violation
        schema = IraqiCoachAdviceSchema.model_validate_json(raw_content)

        return ScamInterventionResponse(
            headline=schema.headline,
            what_looks_wrong=schema.what_looks_wrong,
            scammer_next_move=schema.scammer_next_move,
            recommended_action=schema.recommended_action,
            confidence_level=schema.confidence_level,
            is_fallback=False,
            raw_json=raw_content,
        )

    @staticmethod
    def _fallback_response(reason: str) -> ScamInterventionResponse:
        """
        Returns a pre-verified, Pydantic-compliant fallback response.
        Used whenever any error occurs at Layer 2 so the wallet UI is never blocked.
        """
        logger.warning("Returning verified fallback response | reason=%s", reason)
        return ScamInterventionResponse(
            headline="دير بالك – وقّف التحويل هسه!",
            what_looks_wrong=(
                "التحويل بيه إشارات مريبة حسب قواعدنا، ممكن يكون نصب."
            ),
            scammer_next_move=(
                "النصاب هسه يضغط عليك تسرع وتحول الفلوس قبل ما تفكر – هذا أسلوبهم دايماً."
            ),
            recommended_action=(
                "وقّف التحويل، اقفل الهاتف، واتصل بخدمة عملاء المحفظة مباشرة من الرقم الرسمي."
            ),
            confidence_level=0.9,
            is_fallback=True,
            raw_json="",
        )


# ──────────────────────────────────────────────────────────────────────────────
# Self-test (run directly: python ai_coach.py)
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import pprint

    if hasattr(__import__("sys").stdout, "reconfigure"):
        __import__("sys").stdout.reconfigure(encoding="utf-8", errors="replace")

    coach = IraqiAICoach()
    res = coach.generate_advice(
        amount_iqd=75_000,
        recipient_phone="07801234567",
        note="رسوم استلام الجائزة",
        triggered_rules=["RULE_03_KNOWN_SCAM_FEE"],
        risk_score=0.45,
    )

    print("\n" + "=" * 60)
    print("  [*] VERIFIED LLM ADVICE RESPONSE")
    print("=" * 60)
    pprint.pprint(res.to_dict(), width=70, sort_dicts=False)
