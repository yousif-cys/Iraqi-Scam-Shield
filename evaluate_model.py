"""
Pre-Transaction Anti-Scam Intervention Agent
Held-out Evaluation Benchmark Script

Evaluates Layer 1 Risk Engine on synthetic dataset (transactions.json).
Outputs Precision, Recall, F1 Score, and False Positive Rate (FPR).
"""

import json
from datetime import datetime
from typing import List, Dict, Any
from risk_engine import RiskEngine, TransactionRequest, HistoricalTransaction


def load_dataset(filepath: str = "transactions.json"):
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["users"], data["transactions"]


def evaluate():
    print("=" * 65)
    print("Running Layer 1 Deterministic Risk Engine Benchmark Evaluation...")
    print("=" * 65)

    users, transactions = load_dataset()
    engine = RiskEngine(catalogue_path="scam_catalogue.json")

    # Group transactions by user
    user_tx_map: Dict[str, List[Dict[str, Any]]] = {}
    for tx in transactions:
        uid = tx["user_id"]
        if uid not in user_tx_map:
            user_tx_map[uid] = []
        user_tx_map[uid].append(tx)

    # Confusion Matrix Counters
    tp = 0  # True Positive: Is Scam & Intervened
    fp = 0  # False Positive: Normal & Intervened
    tn = 0  # True Negative: Normal & Not Intervened
    fn = 0  # False Negative: Is Scam & Not Intervened

    scam_breakdown = {}

    for uid, tx_list in user_tx_map.items():
        # Sort chronologically
        tx_list.sort(key=lambda x: datetime.fromisoformat(x["timestamp"]))

        history: List[HistoricalTransaction] = []

        for tx in tx_list:
            tx_time = datetime.fromisoformat(tx["timestamp"])
            is_ground_truth_scam = tx.get("is_scam", False)
            scam_type = tx.get("scam_type", "NORMAL")

            req = TransactionRequest(
                transaction_id=tx["transaction_id"],
                user_id=uid,
                sender_phone=tx["sender_phone"],
                recipient_id=tx["recipient_id"],
                recipient_phone=tx["recipient_phone"],
                recipient_name=tx["recipient_name"],
                amount_iqd=tx["amount_iqd"],
                timestamp=tx_time,
                is_new_recipient=tx["is_new_recipient"],
                transfer_purpose=tx.get("transfer_purpose"),
                user_notes=tx.get("user_notes")
            )

            result = engine.evaluate(req, history)
            intervened = result.requires_intervention

            if is_ground_truth_scam:
                if scam_type not in scam_breakdown:
                    scam_breakdown[scam_type] = {"total": 0, "detected": 0}
                scam_breakdown[scam_type]["total"] += 1

                if intervened:
                    tp += 1
                    scam_breakdown[scam_type]["detected"] += 1
                else:
                    fn += 1
            else:
                if intervened:
                    fp += 1
                else:
                    tn += 1

            # Append to point-in-time history
            history.append(
                HistoricalTransaction(
                    transaction_id=tx["transaction_id"],
                    amount_iqd=tx["amount_iqd"],
                    timestamp=tx_time,
                    recipient_phone=tx["recipient_phone"],
                    is_new_recipient=tx["is_new_recipient"]
                )
            )

    total = tp + fp + tn + fn
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    accuracy = (tp + tn) / total if total > 0 else 0.0

    print(f"Total Evaluated Transactions: {total}")
    print(f"True Positives (Scams Stopped):       {tp}")
    print(f"True Negatives (Legit Uninterrupted): {tn}")
    print(f"False Positives (False Alarms):        {fp}")
    print(f"False Negatives (Missed Scams):        {fn}")
    print("-" * 65)
    print(f"Recall (Scam Catch Rate):              {recall * 100:.2f}%")
    print(f"Precision:                             {precision * 100:.2f}%")
    print(f"F1-Score:                              {f1 * 100:.2f}%")
    print(f"False Positive Rate (FPR):             {fpr * 100:.2f}%")
    print(f"Overall Accuracy:                      {accuracy * 100:.2f}%")
    print("-" * 65)
    print("Detection Breakdown by Iraqi Scam Scenario:")
    for stype, stats in scam_breakdown.items():
        det_rate = (stats["detected"] / stats["total"]) * 100 if stats["total"] > 0 else 0
        print(f"  • {stype}: {stats['detected']}/{stats['total']} ({det_rate:.1f}%)")
    print("=" * 65)

    benchmark_results = {
        "metrics": {
            "total_transactions": total,
            "true_positives": tp,
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
            "recall": round(recall, 4),
            "precision": round(precision, 4),
            "f1_score": round(f1, 4),
            "false_positive_rate": round(fpr, 4),
            "accuracy": round(accuracy, 4)
        },
        "scam_breakdown": scam_breakdown,
        "evaluated_at": datetime.now().isoformat()
    }

    with open("benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(benchmark_results, f, ensure_ascii=False, indent=2)
    print("Benchmark results saved to -> benchmark_results.json")


if __name__ == "__main__":
    evaluate()
