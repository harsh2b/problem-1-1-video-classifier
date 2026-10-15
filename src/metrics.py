"""Evaluation metrics: accuracy, F1, PEAS score, bootstrap CIs, and latency profiling."""
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

from src.config import CLASS_NAMES

def compute_standard_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, Any]:
    """Standard classification metrics."""
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))

    # Per-class breakdown
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    class_report = {
        CLASS_NAMES[i]: round(float(per_class_f1[i]), 4)
        for i in range(len(CLASS_NAMES))
    }

    return {
        "accuracy": round(float(acc), 4),
        "macro_f1": round(float(macro_f1), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "macro_precision": round(float(precision), 4),
        "macro_recall": round(float(recall), 4),
        "per_class_f1": class_report,
        "confusion_matrix": cm.tolist()
    }

def compute_bootstrap_confidence_intervals(
    y_true: List[int], 
    y_pred: List[int], 
    n_bootstraps: int = 1000, 
    confidence_level: float = 0.95
) -> Dict[str, Tuple[float, float]]:
    """Bootstrap 95% confidence intervals for accuracy and macro-F1."""
    np.random.seed(42)
    n = len(y_true)
    boot_accs = []
    boot_f1s = []

    for _ in range(n_bootstraps):
        indices = np.random.choice(n, size=n, replace=True)
        sample_true = [y_true[i] for i in indices]
        sample_pred = [y_pred[i] for i in indices]
        boot_accs.append(accuracy_score(sample_true, sample_pred))
        boot_f1s.append(f1_score(sample_true, sample_pred, average="macro", zero_division=0))

    alpha = (1.0 - confidence_level) / 2.0
    acc_lower = float(np.percentile(boot_accs, alpha * 100))
    acc_upper = float(np.percentile(boot_accs, (1.0 - alpha) * 100))
    f1_lower = float(np.percentile(boot_f1s, alpha * 100))
    f1_upper = float(np.percentile(boot_f1s, (1.0 - alpha) * 100))

    return {
        "accuracy_ci": (round(acc_lower, 4), round(acc_upper, 4)),
        "macro_f1_ci": (round(f1_lower, 4), round(f1_upper, 4))
    }

def compute_peas_score(
    macro_f1: float, 
    latency_p95_ms: float, 
    param_count: int
) -> float:
    """PEAS = Macro_F1^2 / (log10(p95_ms + 1) * log2(Params_M + 1)). Balances accuracy vs cost."""
    params_in_m = param_count / 1_000_000.0
    latency_term = np.log10(max(1.0, latency_p95_ms) + 1.0)
    footprint_term = np.log2(params_in_m + 1.0) + 0.1 # regularizer

    peas = (macro_f1 ** 2) / (latency_term * footprint_term)
    return round(float(peas), 4)

class LatencyProfiler:
    """Simple latency tracker."""
    def __init__(self):
        self.latencies_ms: List[float] = []

    def record(self, duration_sec: float):
        self.latencies_ms.append(duration_sec * 1000.0)

    def summary(self) -> Dict[str, float]:
        if not self.latencies_ms:
            return {"p50_ms": 0.0, "p95_ms": 0.0, "mean_ms": 0.0, "throughput_fps": 0.0}
        arr = np.array(self.latencies_ms)
        p50 = float(np.percentile(arr, 50))
        p95 = float(np.percentile(arr, 95))
        mean = float(np.mean(arr))
        throughput = 1000.0 / mean if mean > 0 else 0.0
        return {
            "p50_ms": round(p50, 2),
            "p95_ms": round(p95, 2),
            "mean_ms": round(mean, 2),
            "throughput_fps": round(throughput, 2)
        }

if __name__ == "__main__":
    # Test metrics module
    y_t = [0, 1, 2, 0, 1, 2, 3, 4, 5]
    y_p = [0, 1, 2, 0, 1, 1, 3, 4, 5]
    std = compute_standard_metrics(y_t, y_p)
    ci = compute_bootstrap_confidence_intervals(y_t, y_p, n_bootstraps=200)
    peas = compute_peas_score(std["macro_f1"], latency_p95_ms=18.5, param_count=85000)
    print("Standard:", std)
    print("95% CI:", ci)
    print("PEAS Score:", peas)
