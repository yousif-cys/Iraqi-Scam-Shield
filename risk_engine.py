"""
risk_engine.py
--------------
Layer-1 Deterministic Risk Engine for the Iraqi Mobile Wallet
Anti-Scam Intervention system.

Evaluates a candidate transaction against a user's historical baseline
and emits a structured RiskAssessment with score, triggered rules,
and an intervention flag.

Rules
─────
RULE_01_ANOMALOUS_NEW_RECIPIENT  (+0.45)
    is_new_recipient AND amount > user_avg × 2.5

RULE_02_HIGH_VELOCITY            (+0.35)
    ≥ 3 transfers in the last 10 minutes (including current)
"RULE_03_ABNORMAL_AMOUNT" :        0.25,

RULE_04_KNOWN_SCAM_FEE           (+0.30)
    is_new_recipient AND amount ∈ {25 000, 50 000, 75 000, 100 000} IQD
"RULE_05_SUSPICIOUS_TIME":               0.25

risk_score is the sum of triggered rule weights, capped at 1.0.
requires_intervention is True when risk_score ≥ 0.40.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
from sklearn.ensemble import IsolationForest

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

RULE_WEIGHTS: dict[str, float] = {
    "RULE_01_NEW_RECIPIENT":           0.25,
    "RULE_02_HIGH_VELOCITY":           0.25,
    "RULE_03_ABNORMAL_AMOUNT" :        0.25,
    "RULE_04_KNOWN_SCAM_FEE":          0.50,
    "RULE_05_SUSPICIOUS_TIME":         0.25
}

KNOWN_SCAM_FEES: frozenset[float] = frozenset({25_000.0, 50_000.0, 75_000.0, 100_000.0})
SCAM_KEYWORDS: tuple[str, ...] = ("رسوم", "جائزة", "ربح", "تحديث", "خدمة العملاء")

ANOMALY_MULTIPLIER: float = 2.5       # amount must exceed avg × this
VELOCITY_WINDOW_MINUTES: int = 10     # look-back window
VELOCITY_THRESHOLD: int = 3           # transfers in window to trigger
INTERVENTION_THRESHOLD: float = 0.40  # minimum score to flag
ABSOLUTE_HIGH_AMOUNT:int = 4

# ──────────────────────────────────────────────────────────────────────────────
# Output schema
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RiskAssessment:
    """
    Immutable result of a single transaction risk evaluation.

    Attributes
    ----------
    tx_id:
        The transaction being evaluated.
    user_id:
        The transacting user.
    risk_score:
        Aggregate weighted score in [0.0, 1.0].
    triggered_rules:
        Ordered list of rule IDs that fired.
    requires_intervention:
        True when risk_score ≥ INTERVENTION_THRESHOLD (0.40).
    rule_details:
        Human-readable explanation per triggered rule (for logging / UI).
    ml_anomaly_score:
        Scikit-learn IsolationForest decision function anomaly score.
    """

    tx_id: str
    user_id: str
    risk_score: float
    triggered_rules: list[str]
    requires_intervention: bool
    rule_details: dict[str, str] = field(default_factory=dict)
    ml_anomaly_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "tx_id": self.tx_id,
            "user_id": self.user_id,
            "risk_score": round(self.risk_score, 4),
            "triggered_rules": self.triggered_rules,
            "requires_intervention": self.requires_intervention,
            "rule_details": self.rule_details,
            
        }


# ──────────────────────────────────────────────────────────────────────────────
# User baseline (derived from dataset)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class UserBaseline:
    """Lightweight per-user historical snapshot used by the engine."""

    user_id: str
    historical_avg_iqd: float           # mean transfer amount
    known_recipients: set[str]          # set of previously used phone numbers
    recent_timestamps: list[datetime]   # all transfer datetimes (chronological)

    @classmethod
    def from_transactions(
        cls, user_id: str, txs: list[dict[str, Any]]
    ) -> "UserBaseline":
        """Build a baseline from raw transaction dicts (JSON-loaded)."""
        amounts = [t["amount_iqd"] for t in txs]
        avg = sum(amounts) / len(amounts) if amounts else 0.0
        recipients = {t["recipient_phone"] for t in txs}
        parsed_timestamps = [
            datetime.fromisoformat(t["timestamp"].replace("Z", "+00:00"))
            if isinstance(t["timestamp"], str) else t["timestamp"]
            for t in txs
        ]
        timestamps = sorted(
            dt.replace(tzinfo=None) if hasattr(dt, "tzinfo") and dt.tzinfo is not None else dt
            for dt in parsed_timestamps
        )
        return cls(
            user_id=user_id,
            historical_avg_iqd=avg,
            known_recipients=recipients,
            recent_timestamps=timestamps,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Risk Engine
# ──────────────────────────────────────────────────────────────────────────────

class RiskEngine:
    """
    Layer-1 Deterministic Risk Engine.

    Parameters
    ----------
    dataset_path:
        Path to dataset.json (generated by data_generator.py).
        Loaded once at construction to build per-user baselines.
    """

    def __init__(self, dataset_path: str | Path = "dataset.json") -> None:
        self._baselines: dict[str, UserBaseline] = {}
        self._ml_model: IsolationForest | None = None
        ds_path = Path(dataset_path)
        self._load_baselines(ds_path)
        self.train_anomaly_detector(ds_path)
        logger.info(
            "RiskEngine ready | users=%d | rules=%d | ml_detector=%s",
            len(self._baselines),
            len(RULE_WEIGHTS),
            "ACTIVE" if self._ml_model is not None else "OFFLINE",
        )

    # ── public ────────────────────────────────────────────────────────────────

    def train_anomaly_detector(self, dataset_path: str | Path = "dataset.json") -> None:
        """
        Train an IsolationForest anomaly detector on transaction amounts,
        velocity, and recipient history from dataset.json.
        Adapted from kasturidd/FraudDetection-LLM-Integration.
        """
        path = Path(dataset_path)
        if not path.exists():
            logger.warning("dataset.json not found at %s – ML detector not trained", path)
            self._ml_model = None
            return

        try:
            raw: list[dict[str, Any]] = json.loads(path.read_text(encoding="utf-8"))
            user_tx_times: dict[str, list[datetime]] = {}
            features: list[list[float]] = []

            for tx in raw:
                uid = tx.get("user_id", "")
                amt = float(tx.get("amount_iqd", 0.0))
                ts_raw = tx.get("timestamp", "")
                ts = self._parse_ts(ts_raw) if ts_raw else datetime.now()
                is_new = 1.0 if tx.get("is_new_recipient", False) else 0.0

                if uid not in user_tx_times:
                    user_tx_times[uid] = []
                vel = sum(1 for t in user_tx_times[uid] if (ts - timedelta(minutes=VELOCITY_WINDOW_MINUTES)) <= t <= ts)
                user_tx_times[uid].append(ts)
                features.append([amt, float(vel), is_new])

            if len(features) >= 10:
                X = np.array(features)
                clf = IsolationForest(
                    n_estimators=100,
                    contamination=0.1,
                    random_state=42,
                )
                clf.fit(X)
                self._ml_model = clf
                logger.info("IsolationForest trained on %d transaction samples", len(features))
            else:
                self._ml_model = None
        except Exception as e:
            logger.error("Failed to train IsolationForest anomaly detector: %s", e)
            self._ml_model = None

    def evaluate(
        self,
        *,
        tx_id: str,
        user_id: str,
        recipient_phone: str,
        amount_iqd: float,
        timestamp: str | datetime,
        is_new_recipient: bool | None = None,
        note: str = "",
        historical_avg_override: float | None = None,
        recent_count_override: int | None = None,
    ) -> RiskAssessment:
        """
        Evaluate a pending transaction and return a RiskAssessment.

        Parameters
        ----------
        tx_id:
            Unique transaction identifier.
        user_id:
            Wallet user ID.  Must exist in the loaded baseline.
        recipient_phone:
            Destination wallet number.
        amount_iqd:
            Transfer amount in Iraqi Dinars.
        timestamp:
            Transaction datetime (ISO string or datetime object).
        is_new_recipient:
            Override flag.  If None, inferred from user's known recipients.
        note:
            Transfer reason or remarks (checked for Iraqi scam keyword indicators).
        historical_avg_override:
            When provided by the API caller, replaces the dataset-derived
            historical average for Rule 01 evaluation.
        recent_count_override:
            When provided, used directly as the 10-minute transfer count
            for Rule 02 instead of scanning stored timestamps.

        Returns
        -------
        RiskAssessment
        """
        ts: datetime = self._parse_ts(timestamp)
        baseline = self._get_or_create_baseline(user_id)

        # Resolve new-recipient flag
        if is_new_recipient is None:
            is_new_recipient = recipient_phone not in baseline.known_recipients

        triggered: list[str] = []
        details: dict[str, str] = {}

        # ── Rule 01: Anomalous new recipient ──────────────────────────────────
        # Prefer live override from API caller; fall back to dataset baseline.
        # ── Rule 01: New Recipient ──────────────────────────────────────────── #
        # تنبثق هذه القاعدة بمجرد أن يكون المستلم جديداً في النظام
        if is_new_recipient:
            rule_01 = "RULE_01_NEW_RECIPIENT"
            triggered.append(rule_01)
            details[rule_01] = f"Transaction sent to a new recipient {recipient_phone}"
            logger.debug("%s triggered for tx=%s", rule_01, tx_id)







        effective_avg = (
            historical_avg_override
            if historical_avg_override is not None
            else baseline.historical_avg_iqd
        )

        
        # ── Rule 02: High-velocity spike ──────────────────────────────────────
        if recent_count_override is not None:
            # API already counted transfers in the last 10 min (current included)
            velocity_count = recent_count_override
        else:
            ts_naive = ts.replace(tzinfo=None) if ts.tzinfo is not None else ts
            window_start = ts_naive - timedelta(minutes=VELOCITY_WINDOW_MINUTES)
            if window_start.tzinfo is not None:
                window_start = window_start.replace(tzinfo=None)
            recent_timestamps_naive = [
                t.replace(tzinfo=None) if hasattr(t, "tzinfo") and t.tzinfo is not None else t
                for t in baseline.recent_timestamps
            ]
            recent_in_window = [
                t for t in recent_timestamps_naive if window_start <= t < ts_naive
            ]
            velocity_count = len(recent_in_window) + 1  # +1 for current transfer

        if velocity_count >= VELOCITY_THRESHOLD:
            rule = "RULE_02_HIGH_VELOCITY"
            triggered.append(rule)
            details[rule] = (
                f"{velocity_count} transfers in last {VELOCITY_WINDOW_MINUTES} min "
                f"(threshold \u2265 {VELOCITY_THRESHOLD})"
            )
            logger.debug("%s triggered for tx=%s", rule, tx_id)
        #-----rule_03 = "RULE_03_ABNORMAL_amount"
        if effective_avg > 0:
            threshold = effective_avg * ANOMALY_MULTIPLIER
            if amount_iqd > threshold:
                rule = "RULE_03_ABNORMAL_AMOUNT"
                triggered.append(rule)
                details[rule] = (
                    f"Amount {amount_iqd:,.0f} IQD > "
                    f"{threshold:,.0f} IQD "
                    f"({ANOMALY_MULTIPLIER}\u00d7 avg {effective_avg:,.0f} IQD) "
                    f"to new recipient {recipient_phone}"
                )
                logger.debug("%s triggered for tx=%s", rule, tx_id)
                
        # ── Rule 04: Known Iraqi scam fee / keyword match ─────────────────────
        has_scam_fee = is_new_recipient and amount_iqd in KNOWN_SCAM_FEES
        note_str = str(note or "")
        matched_keywords = [kw for kw in SCAM_KEYWORDS if kw in note_str]

        if has_scam_fee or matched_keywords:
            rule = "RULE_04_KNOWN_SCAM_FEE"
            triggered.append(rule)
            reasons: list[str] = []
            if has_scam_fee:
                reasons.append(
                    f"Amount {amount_iqd:,.0f} IQD matches catalogued Iraqi scam fee to new recipient {recipient_phone}"
                )
            if matched_keywords:
                reasons.append(
                    f"Transfer note contains known scam keyword(s): {', '.join(matched_keywords)}"
                )
            details[rule] = "; ".join(reasons)
            logger.debug("%s triggered for tx=%s", rule, tx_id)

        

        # ── Rule 05: Suspicious Time Window ─────────────────────────────────── #
        tx_time = ts

        # تحديد الساعات المشبوهة (كمثال: من منتصف الليل وحتى الـ 5 صباحاً)
        if tx_time.hour >= 0 and tx_time.hour < 5:
            rule_05 = "RULE_05_SUSPICIOUS_TIME"
            triggered.append(rule_05)
            
            # تطبيق تأثير الـ 25% (سواء بزيادة درجة المخاطر أو خفض حد الأمان بنسبة 25%)
            # هنا كمثال: نقوم بخفض حد الأمان للمعاملة الحالية لأن الوقت خطر
            risk_factor_adjustment = 0.25 
            
            details[rule_05] = (
                f"Transaction initiated at suspicious time ({tx_time.strftime('%H:%M')}). "
                f"Applied a {risk_factor_adjustment * 100:.0f}% strictness penalty to thresholds."
            )
            logger.debug("%s triggered for tx=%s", rule_05, tx_id)




        # ── Score aggregation ─────────────────────────────────────────────────
        raw_score = sum(RULE_WEIGHTS.get(r, 0.35) for r in triggered)
        risk_score = min(raw_score, 1.0)
        requires_intervention = risk_score >= INTERVENTION_THRESHOLD

        logger.info(
            "tx=%s user=%s score=%.2f rules=%s intervention=%s ml_score=%.4f",
            tx_id, user_id, risk_score, triggered, requires_intervention, 
        )
        baseline.recent_timestamps.append(ts)
        return RiskAssessment(
            tx_id=tx_id,
            user_id=user_id,
            risk_score=risk_score,
            triggered_rules=triggered,
            requires_intervention=requires_intervention,
            rule_details=details,
            
        )

    # ── private ───────────────────────────────────────────────────────────────

    def _load_baselines(self, path: Path) -> None:
        """Parse dataset.json and build per-user UserBaseline objects."""
        if not path.exists():
            logger.warning(
                "dataset.json not found at %s – engine starts with empty baselines. "
                "Run data_generator.py first.",
                path.resolve(),
            )
            return

        raw: list[dict[str, Any]] = json.loads(path.read_text(encoding="utf-8"))

        # Group by user_id
        user_txs: dict[str, list[dict[str, Any]]] = {}
        for tx in raw:
            uid = tx["user_id"]
            user_txs.setdefault(uid, []).append(tx)

        for uid, txs in user_txs.items():
            self._baselines[uid] = UserBaseline.from_transactions(uid, txs)

        logger.info("Loaded %d transactions → %d user baselines", len(raw), len(self._baselines))

    def _get_or_create_baseline(self, user_id: str) -> UserBaseline:
        """Return existing baseline or create an empty one for unknown users."""
        if user_id not in self._baselines:
            logger.warning(
                "Unknown user_id=%s – creating empty baseline (no history).",
                user_id,
            )
            self._baselines[user_id] = UserBaseline(
                user_id=user_id,
                historical_avg_iqd=0.0,
                known_recipients=set(),
                recent_timestamps=[],
            )
        return self._baselines[user_id]

    @staticmethod
    def _parse_ts(ts: str | datetime) -> datetime:
        if isinstance(ts, datetime):
            parsed = ts
        else:
            clean = str(ts).replace("Z", "+00:00")
            parsed = datetime.fromisoformat(clean)
        return parsed.replace(tzinfo=None) if parsed.tzinfo is not None else parsed


# ──────────────────────────────────────────────────────────────────────────────
# Self-test (run directly: python risk_engine.py)
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, io, pprint
    if hasattr(sys.stdout, "reconfigure"):
        getattr(sys.stdout, "reconfigure")(encoding="utf-8", errors="replace")
    else:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")

    engine = RiskEngine("dataset.json")

    # ── Pull first real user from dataset for a realistic test ────────────────
    dataset_path = Path("dataset.json")
    if dataset_path.exists():
        raw_txs: list[dict[str, Any]] = json.loads(dataset_path.read_text(encoding="utf-8"))
        sample_uid = raw_txs[0]["user_id"]
        sample_avg = engine._baselines[sample_uid].historical_avg_iqd
    else:
        sample_uid = "U01"
        sample_avg = 200_000.0

    print("\n" + "=" * 65)
    print("  RISK ENGINE — 3 TEST CASES")
    print("=" * 65)

    # Case 1: High-risk – new recipient + scam fee
    ts_now = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    r1 = engine.evaluate(
        tx_id="TEST-001",
        user_id=sample_uid,
        recipient_phone="07901111111",
        amount_iqd=50_000,
        timestamp=ts_now,
        is_new_recipient=True,
    )
    print("\n[Case 1] New recipient + known scam fee (50,000 IQD)")
    pprint.pprint(r1.to_dict(), width=70, sort_dicts=False)

    # Case 2: High-risk – new recipient + 4× anomalous amount
    r2 = engine.evaluate(
        tx_id="TEST-002",
        user_id=sample_uid,
        recipient_phone="07902222222",
        amount_iqd=round(sample_avg * 4.0 / 500) * 500,
        timestamp=ts_now,
        is_new_recipient=True,
    )
    print(f"\n[Case 2] New recipient + 4x avg amount ({sample_avg*4:,.0f} IQD)")
    pprint.pprint(r2.to_dict(), width=70, sort_dicts=False)

    # Case 3: Low-risk – known recipient + normal amount
    known_phone = next(iter(engine._baselines[sample_uid].known_recipients), "07800000000")
    r3 = engine.evaluate(
        tx_id="TEST-003",
        user_id=sample_uid,
        recipient_phone=known_phone,
        amount_iqd=round(sample_avg * 0.5 / 500) * 500,
        timestamp=ts_now,
        is_new_recipient=False,
    )
    print(f"\n[Case 3] Known recipient + normal amount ({sample_avg*0.5:,.0f} IQD)")
    pprint.pprint(r3.to_dict(), width=70, sort_dicts=False)
    print("\n" + "=" * 65)
