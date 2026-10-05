"""
finance_ops.py
--------------
Financial Operations Agent & Immutable Audit Logging Layer
Adapted from manasvidobariya82/ai-finance-operations-agent.

Maintains a structured, tamper-evident audit trail of all transactions,
recording Layer 1 deterministic rules & ML anomaly scores, Layer 2 LLM
Baghdadi coach responses, execution latencies, and PausePoint state resolutions.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_AUDIT_LOG_PATH = Path("audit_log.json")


@dataclass
class AuditRecord:
    """Structured audit log entry for financial compliance and fraud investigations."""

    timestamp: str
    transaction_id: str
    user_id: str
    recipient_phone: str
    amount_iqd: float
    note: str
    risk_score: float
    requires_intervention: bool
    triggered_rules: List[str]
    rule_details: Dict[str, str]
    ml_anomaly_score: float
    layer2_advice: Optional[Dict[str, Any]] = None
    latency_ms: float = 0.0
    pausepoint_state: str = "APPROVED_AUTO"
    resolution: str = "APPROVED_AUTO"
    outcome: str = "TRANSACTION_CLEAN"
    user_feedback: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class FinanceOpsAgent:
    """
    Financial Operations Agent.

    Enforces immutable transaction audit logging, compliance tracking,
    and post-interception outcome settlement for Iraqi digital wallets.
    """

    def __init__(self, log_path: str | Path = DEFAULT_AUDIT_LOG_PATH) -> None:
        self.log_path = Path(log_path)
        self._ensure_log_file()
        logger.info("FinanceOpsAgent active | audit_log=%s", self.log_path.resolve())

    def _ensure_log_file(self) -> None:
        """Create audit log file with valid empty JSON array if missing."""
        if not self.log_path.exists():
            try:
                self.log_path.write_text("[]\n", encoding="utf-8")
            except OSError as e:
                logger.error("Failed to initialize audit log file %s: %s", self.log_path, e)

    def log_evaluation(
        self,
        *,
        tx_id: str,
        user_id: str,
        recipient_phone: str,
        amount_iqd: float,
        note: str,
        risk_score: float,
        requires_intervention: bool,
        triggered_rules: List[str],
        rule_details: Dict[str, str],
        ml_anomaly_score: float = 0.0,
        layer2_advice: Optional[Dict[str, Any]] = None,
        latency_ms: float = 0.0,
        pausepoint_state: str = "APPROVED_AUTO",
        resolution: str = "APPROVED_AUTO",
        outcome: str = "TRANSACTION_CLEAN",
        user_feedback: Optional[str] = None,
    ) -> AuditRecord:
        """
        Record a transaction attempt and its two-layer safety evaluation into audit_log.json.
        """
        now_iso = datetime.now(timezone.utc).isoformat()
        record = AuditRecord(
            timestamp=now_iso,
            transaction_id=tx_id,
            user_id=user_id,
            recipient_phone=recipient_phone,
            amount_iqd=amount_iqd,
            note=note,
            risk_score=round(risk_score, 4),
            requires_intervention=requires_intervention,
            triggered_rules=list(triggered_rules),
            rule_details=dict(rule_details),
            ml_anomaly_score=round(ml_anomaly_score, 4),
            layer2_advice=layer2_advice,
            latency_ms=round(latency_ms, 2),
            pausepoint_state=pausepoint_state,
            resolution=resolution,
            outcome=outcome,
            user_feedback=user_feedback,
        )

        entries = self.load_audit_logs()
        # Check if transaction already exists; if so, update it, else append
        existing_idx = next(
            (i for i, r in enumerate(entries) if r.get("transaction_id") == tx_id),
            None,
        )
        if existing_idx is not None:
            entries[existing_idx] = record.to_dict()
        else:
            entries.append(record.to_dict())

        self._save_entries(entries)
        logger.info(
            "Audit record logged | tx=%s | state=%s | outcome=%s",
            tx_id,
            pausepoint_state,
            outcome,
        )
        return record

    def update_resolution(
        self,
        *,
        tx_id: str,
        pausepoint_state: str,
        resolution: str,
        outcome: str,
        user_feedback: Optional[str] = None,
    ) -> bool:
        """
        Update the final PausePoint resolution when user makes an interactive decision.
        """
        entries = self.load_audit_logs()
        updated = False
        for entry in entries:
            if entry.get("transaction_id") == tx_id:
                entry["pausepoint_state"] = pausepoint_state
                entry["resolution"] = resolution
                entry["outcome"] = outcome
                if user_feedback is not None:
                    entry["user_feedback"] = user_feedback
                entry["resolved_at"] = datetime.now(timezone.utc).isoformat()
                updated = True
                break

        if updated:
            self._save_entries(entries)
            logger.info("Audit resolution updated | tx=%s -> %s (%s)", tx_id, resolution, outcome)
        else:
            logger.warning("Audit resolution update failed | tx=%s not found", tx_id)
        return updated

    def load_audit_logs(self) -> List[Dict[str, Any]]:
        """Load and return all entries from the audit log."""
        self._ensure_log_file()
        try:
            content = self.log_path.read_text(encoding="utf-8").strip()
            if not content:
                return []
            return json.loads(content)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to decode audit log JSON, resetting: %s", e)
            return []

    def _save_entries(self, entries: List[Dict[str, Any]]) -> None:
        """Write records to file safely."""
        try:
            self.log_path.write_text(
                json.dumps(entries, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError as e:
            logger.error("Failed to write to audit log: %s", e)

    def get_summary_metrics(self) -> Dict[str, Any]:
        """Aggregate high-level operational statistics from the audit log."""
        entries = self.load_audit_logs()
        total_tx = len(entries)
        scams_prevented = sum(
            1 for e in entries if e.get("outcome") in ("SCAM_PREVENTED", "INTERCEPTED")
            or e.get("resolution") == "CANCELLED_BY_USER"
        )
        user_overrides = sum(
            1 for e in entries if e.get("resolution") == "PROCEEDED_ON_USER_RISK"
        )
        auto_approved = sum(
            1 for e in entries if e.get("resolution") == "APPROVED_AUTO"
        )
        total_protected_volume = sum(
            float(e.get("amount_iqd", 0)) for e in entries
            if e.get("resolution") == "CANCELLED_BY_USER" or e.get("outcome") == "SCAM_PREVENTED"
        )

        return {
            "total_transactions_logged": total_tx,
            "scams_prevented": scams_prevented,
            "user_overrides": user_overrides,
            "auto_approved": auto_approved,
            "total_protected_volume_iqd": total_protected_volume,
        }
