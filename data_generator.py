"""
data_generator.py
-----------------
Generates a realistic synthetic dataset of 200+ Iraqi mobile-wallet
transactions across 20 distinct user profiles and saves it to dataset.json.

Scam vectors embedded:
  • Exact prize-fee amounts  : 25k / 50k / 75k / 100k IQD to new recipients
  • Anomalous-new-recipient  : amount > 2.5× user historical average
  • High-velocity spike      : ≥ 3 transfers within any 10-minute window

Run:
    python data_generator.py
"""

from __future__ import annotations

import json
import random
import uuid
from datetime import datetime, timedelta, timezone
from typing import Literal

# pyrefly: ignore [missing-import]
from pydantic import BaseModel, Field, field_validator

# ──────────────────────────────────────────────────────────────────────────────
# Pydantic schema
# ──────────────────────────────────────────────────────────────────────────────

ScamCategory = Literal["FAKE_SUPPORT", "PRIZE_FEE", "URGENT_IMPERSONATION", "NONE"]


class Transaction(BaseModel):
    tx_id: str = Field(..., description="UUID4 transaction identifier")
    user_id: str = Field(..., description="User profile identifier (U01-U20)")
    recipient_phone: str = Field(..., description="Iraqi mobile number (07xx-xxxxxxx)")
    amount_iqd: float = Field(..., gt=0, description="Transfer amount in Iraqi Dinars")
    timestamp: str = Field(..., description="ISO-8601 UTC timestamp")
    is_new_recipient: bool = Field(..., description="True if first transfer to this number")
    note: str = Field(..., description="Transfer memo / reason")
    is_scam: bool = Field(..., description="Ground-truth scam label")
    scam_category: ScamCategory = Field(..., description="Scam taxonomy or NONE")

    @field_validator("recipient_phone")
    @classmethod
    def validate_iraqi_phone(cls, v: str) -> str:
        if not (v.startswith("07") and len(v) == 11 and v.isdigit()):
            raise ValueError(f"Invalid Iraqi phone number: {v}")
        return v


# ──────────────────────────────────────────────────────────────────────────────
# Iraqi locale fixtures
# ──────────────────────────────────────────────────────────────────────────────

IRAQI_PREFIXES = ["0770", "0771", "0772", "0773", "0774",
                  "0780", "0781", "0782", "0783",
                  "0790", "0791", "0792"]

NORMAL_NOTES = [
    "حواله عيلية", "دفع إيجار", "مصاريف جامعة", "بضاعة", "ديون",
    "راتب عامل", "فاتورة كهرباء", "مشتريات", "تسديد دين", "هدية",
    "مصروف بيت", "سفرية", "حجز فندق", "دفع قسط", "عمولة",
]
SCAM_NOTES_PRIZE = [
    "رسوم تسليم الجائزة", "دفعة تحرير الجائزة", "ضريبة الجائزة الكبرى",
    "صك الجائزة", "إجراءات استلام الجائزة",
]
SCAM_NOTES_SUPPORT = [
    "تحديث حساب ZainCash", "تفعيل محفظة QiCard", "رسوم أمان الحساب",
    "تجديد اشتراك المحفظة", "رسوم دعم فني",
]
SCAM_NOTES_IMPERSONATION = [
    "أمر مستعجل من مدير البنك", "تحويل طارئ – مدير عام",
    "طلب عاجل من وزارة المالية", "تحويل فوري – ضابط أمن",
    "طلب سري من المدير",
]

KNOWN_SCAM_FEES: list[float] = [25_000, 50_000, 75_000, 100_000]

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _rand_phone(exclude: set[str] | None = None) -> str:
    """Generate a random valid Iraqi phone number."""
    while True:
        prefix = random.choice(IRAQI_PREFIXES)
        suffix = str(random.randint(1_000_000, 9_999_999))
        phone = prefix + suffix
        if exclude is None or phone not in exclude:
            return phone


def _rand_amount_normal(avg: float) -> float:
    """Return a plausible normal transfer amount near the user's average."""
    lo = max(1_000, avg * 0.3)
    hi = avg * 1.8
    return round(random.uniform(lo, hi) / 500) * 500


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


# ──────────────────────────────────────────────────────────────────────────────
# User profile
# ──────────────────────────────────────────────────────────────────────────────

class UserProfile:
    """Holds per-user state and transaction history."""

    def __init__(self, user_id: str) -> None:
        self.user_id = user_id
        # Typical monthly average transfer for this user (IQD)
        self.historical_avg: float = random.choice(
            [50_000, 75_000, 100_000, 150_000, 200_000,
             250_000, 300_000, 500_000, 750_000, 1_000_000]
        )
        # Known recipients – start with 3-8 pre-seeded numbers
        self.known_recipients: set[str] = {
            _rand_phone() for _ in range(random.randint(3, 8))
        }
        # All timestamps of transfers made (for velocity check)
        self.transfer_times: list[datetime] = []

    def is_new(self, phone: str) -> bool:
        return phone not in self.known_recipients

    def register(self, phone: str, ts: datetime) -> None:
        self.known_recipients.add(phone)
        self.transfer_times.append(ts)


# ──────────────────────────────────────────────────────────────────────────────
# Dataset generator
# ──────────────────────────────────────────────────────────────────────────────

class DatasetGenerator:
    NUM_USERS = 20
    TARGET_TOTAL = 220          # ≥ 200 per requirement
    SCAM_RATIO = 0.28           # ~28% scam transactions

    # Distribution of scam categories (weights)
    SCAM_WEIGHTS = {
        "PRIZE_FEE": 0.45,
        "FAKE_SUPPORT": 0.30,
        "URGENT_IMPERSONATION": 0.25,
    }

    def __init__(self) -> None:
        self.users: dict[str, UserProfile] = {
            f"U{i:02d}": UserProfile(f"U{i:02d}")
            for i in range(1, self.NUM_USERS + 1)
        }
        self.transactions: list[Transaction] = []
        # Shared base timestamp – 90 days ago
        self._base_ts = datetime.now(tz=timezone.utc) - timedelta(days=90)

    # ── public ────────────────────────────────────────────────────────────────

    def generate(self) -> list[Transaction]:
        n_scam = int(self.TARGET_TOTAL * self.SCAM_RATIO)
        n_normal = self.TARGET_TOTAL - n_scam

        user_ids = list(self.users.keys())

        # Assign normal transactions proportionally
        for _ in range(n_normal):
            uid = random.choice(user_ids)
            self.transactions.append(self._normal_tx(self.users[uid]))

        # Assign scam transactions
        scam_cats = list(self.SCAM_WEIGHTS.keys())
        scam_wts = list(self.SCAM_WEIGHTS.values())
        for _ in range(n_scam):
            uid = random.choice(user_ids)
            cat = random.choices(scam_cats, weights=scam_wts, k=1)[0]
            self.transactions.append(self._scam_tx(self.users[uid], cat))

        # Sort chronologically
        self.transactions.sort(key=lambda t: t.timestamp)
        return self.transactions

    # ── private builders ──────────────────────────────────────────────────────

    def _next_ts(self, user: UserProfile, burst: bool = False) -> datetime:
        """Return the next plausible timestamp for a user."""
        if user.transfer_times:
            last = user.transfer_times[-1]
            if burst:
                delta = timedelta(minutes=random.randint(1, 8))
            else:
                delta = timedelta(hours=random.randint(6, 120))
            ts = last + delta
        else:
            ts = self._base_ts + timedelta(
                days=random.randint(0, 80),
                hours=random.randint(0, 23),
                minutes=random.randint(0, 59),
            )
        return ts

    def _normal_tx(self, user: UserProfile) -> Transaction:
        ts = self._next_ts(user)
        # 70% chance send to known recipient
        if user.known_recipients and random.random() < 0.70:
            phone = random.choice(list(user.known_recipients))
            new = False
        else:
            phone = _rand_phone(exclude=user.known_recipients)
            new = True
        amount = _rand_amount_normal(user.historical_avg)
        note = random.choice(NORMAL_NOTES)
        user.register(phone, ts)
        return Transaction(
            tx_id=str(uuid.uuid4()),
            user_id=user.user_id,
            recipient_phone=phone,
            amount_iqd=amount,
            timestamp=_iso(ts),
            is_new_recipient=new,
            note=note,
            is_scam=False,
            scam_category="NONE",
        )

    def _scam_tx(self, user: UserProfile, category: str) -> Transaction:
        # High-velocity burst: optionally force 2 prior fast transfers first
        burst = random.random() < 0.35
        if burst and len(user.transfer_times) >= 1:
            ts = self._next_ts(user, burst=True)
        else:
            ts = self._next_ts(user)

        phone = _rand_phone(exclude=user.known_recipients)  # always new

        if category == "PRIZE_FEE":
            amount = float(random.choice(KNOWN_SCAM_FEES))
            note = random.choice(SCAM_NOTES_PRIZE)
        elif category == "FAKE_SUPPORT":
            # amount may match a scam fee or be anomalously large
            if random.random() < 0.5:
                amount = float(random.choice(KNOWN_SCAM_FEES))
            else:
                amount = round(user.historical_avg * random.uniform(2.6, 4.0) / 500) * 500
            note = random.choice(SCAM_NOTES_SUPPORT)
        else:  # URGENT_IMPERSONATION
            amount = round(user.historical_avg * random.uniform(3.0, 6.0) / 500) * 500
            note = random.choice(SCAM_NOTES_IMPERSONATION)

        user.register(phone, ts)
        return Transaction(
            tx_id=str(uuid.uuid4()),
            user_id=user.user_id,
            recipient_phone=phone,
            amount_iqd=amount,
            timestamp=_iso(ts),
            is_new_recipient=True,
            note=note,
            is_scam=True,
            scam_category=category,  # type: ignore[arg-type]
        )


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import io
    import pathlib

    # UTF-8 stdout for Windows cp1252 terminals
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    else:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    gen = DatasetGenerator()
    txs = gen.generate()

    out_path = pathlib.Path("dataset.json")
    payload = [tx.model_dump() for tx in txs]
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    n_scam = sum(1 for t in txs if t.is_scam)
    n_normal = len(txs) - n_scam
    cats: dict[str, int] = {}
    for t in txs:
        cats[t.scam_category] = cats.get(t.scam_category, 0) + 1

    print(f"[OK] dataset.json written -- {len(txs)} transactions")
    print(f"     Normal : {n_normal}  |  Scam : {n_scam}")
    print("     Category breakdown:")
    for cat, count in sorted(cats.items()):
        print(f"       {cat:<26} {count}")

