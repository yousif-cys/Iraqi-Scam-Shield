"""
tests/test_ai_coach.py
-----------------------
Unit test suite for Layer 1 Risk Engine immutability & Layer 2 AI Coach Safety Gate.

Covers:
1. Valid Groq JSON output parsing & Pydantic schema validation.
2. Malformed JSON handling -> Verified Fallback trigger.
3. Schema validation failure (missing keys / empty strings) -> Verified Fallback trigger.
4. API Timeout / Network Failure -> Verified Fallback trigger.
5. Immutability check on RiskAssessment dataclass.
6. Decision authority isolation (Layer 2 cannot override Layer 1 decisions).

Run:
    python -m unittest tests/test_ai_coach.py
"""

from __future__ import annotations

import sys
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest.mock import MagicMock, patch

# Configure sys.path so local modules are found
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# pyrefly: ignore [missing-import]
from groq import APIConnectionError, APITimeoutError
from ai_coach import IraqiAICoach, IraqiCoachAdviceSchema, ScamInterventionResponse
from risk_engine import RiskAssessment, RiskEngine


class TestIraqiAICoachSafetyLayer(unittest.TestCase):
    """Unit test suite verifying Output Safety Layer and Decision Authority Isolation."""

    def setUp(self) -> None:
        self.coach = IraqiAICoach()

    # -------------------------------------------------------------------------
    # 1. Valid Groq JSON output parsing
    # -------------------------------------------------------------------------
    @patch("ai_coach.groq_client.chat.completions.create")
    def test_valid_groq_json_parsing(self, mock_create: MagicMock) -> None:
        """Verifies that valid Groq JSON response is parsed into ScamInterventionResponse."""
        valid_json = (
            '{"headline": "طوّل بالك عيني، المعاملة مشبوهة", '
            '"scammer_next_move": "راح يطلب منك تحويل الفلوس أو رمز OTP بالهاتف", '
            '"recommended_action": "اقفل الخط فوراً ولا تحوّل أي مبلغ", '
            '"confidence_level": 0.95}'
        )
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message.content = valid_json
        mock_create.return_value = mock_completion

        res = self.coach.evaluate_transaction(
            amount_iqd=50000.0,
            recipient_name="07901234567",
            recipient_account="07901234567",
            transaction_note="رسوم استلام الجائزة",
        )

        self.assertFalse(res.is_fallback)
        self.assertEqual(res.headline, "طوّل بالك عيني، المعاملة مشبوهة")
        self.assertEqual(res.confidence_level, 0.95)

    # -------------------------------------------------------------------------
    # 2. Malformed JSON handling -> Fallback trigger
    # -------------------------------------------------------------------------
    @patch("ai_coach.groq_client.chat.completions.create")
    def test_malformed_json_handling(self, mock_create: MagicMock) -> None:
        """Verifies that malformed JSON triggers verified fallback safely."""
        malformed_json = '{"headline": "طوّل بالك", "scammer_next_move": ... incomplete ...'
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message.content = malformed_json
        mock_create.return_value = mock_completion

        res = self.coach.evaluate_transaction(
            amount_iqd=50000.0,
            recipient_name="07901234567",
            recipient_account="07901234567",
        )

        self.assertTrue(res.is_fallback)
        self.assertIn("دير بالك", res.headline)

    # -------------------------------------------------------------------------
    # 3. Schema validation failure -> Fallback trigger
    # -------------------------------------------------------------------------
    @patch("ai_coach.groq_client.chat.completions.create")
    def test_schema_validation_failure_missing_keys(self, mock_create: MagicMock) -> None:
        """Verifies missing required keys trigger verified fallback."""
        incomplete_json = '{"headline": "تحذير عاجل"}'  # missing scammer_next_move & recommended_action
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message.content = incomplete_json
        mock_create.return_value = mock_completion

        res = self.coach.evaluate_transaction(
            amount_iqd=50000.0,
            recipient_name="07901234567",
            recipient_account="07901234567",
        )

        self.assertTrue(res.is_fallback)

    @patch("ai_coach.groq_client.chat.completions.create")
    def test_schema_validation_failure_empty_strings(self, mock_create: MagicMock) -> None:
        """Verifies empty strings trigger Pydantic validation error -> Fallback."""
        empty_field_json = (
            '{"headline": " ", "scammer_next_move": "تكتيك", "recommended_action": "إجراء"}'
        )
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message.content = empty_field_json
        mock_create.return_value = mock_completion

        res = self.coach.evaluate_transaction(
            amount_iqd=50000.0,
            recipient_name="07901234567",
            recipient_account="07901234567",
        )

        self.assertTrue(res.is_fallback)

    # -------------------------------------------------------------------------
    # 4. API Timeout / Network Failure -> Fallback trigger
    # -------------------------------------------------------------------------
    @patch("ai_coach.groq_client.chat.completions.create")
    def test_api_timeout_failure(self, mock_create: MagicMock) -> None:
        """Verifies APITimeoutError triggers fallback without raising uncaught exceptions."""
        mock_create.side_effect = APITimeoutError(request=MagicMock())

        res = self.coach.evaluate_transaction(
            amount_iqd=50000.0,
            recipient_name="07901234567",
            recipient_account="07901234567",
        )

        self.assertTrue(res.is_fallback)

    @patch("ai_coach.groq_client.chat.completions.create")
    def test_api_connection_failure(self, mock_create: MagicMock) -> None:
        """Verifies APIConnectionError triggers fallback."""
        mock_create.side_effect = APIConnectionError(request=MagicMock())

        res = self.coach.evaluate_transaction(
            amount_iqd=50000.0,
            recipient_name="07901234567",
            recipient_account="07901234567",
        )

        self.assertTrue(res.is_fallback)

    # -------------------------------------------------------------------------
    # 5. Immutability check on RiskAssessment dataclass
    # -------------------------------------------------------------------------
    def test_risk_assessment_immutability(self) -> None:
        """Verifies that RiskAssessment is strictly immutable (frozen dataclass)."""
        assessment = RiskAssessment(
            tx_id="TX-100",
            user_id="U01",
            risk_score=0.85,
            triggered_rules=["RULE_01_ANOMALOUS_NEW_RECIPIENT"],
            requires_intervention=True,
        )

        with self.assertRaises((FrozenInstanceError, AttributeError)):
            # Attempt to mutate requires_intervention field
            assessment.requires_intervention = False  # type: ignore

    # -------------------------------------------------------------------------
    # 6. Decision Authority Isolation
    # -------------------------------------------------------------------------
    def test_decision_authority_isolation(self) -> None:
        """Verifies Layer 2 does not invoke LLM when Layer 1 requires_intervention is False."""
        safe_assessment = RiskAssessment(
            tx_id="TX-200",
            user_id="U01",
            risk_score=0.0,
            triggered_rules=[],
            requires_intervention=False,
        )

        res = self.coach.get_coach_advice(
            assessment=safe_assessment,
            amount_iqd=10000.0,
            recipient_phone="07801234567",
        )

        self.assertTrue(res.is_fallback)


if __name__ == "__main__":
    unittest.main()
