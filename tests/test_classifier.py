"""Tests for the video classifier."""
import pytest
import torch
import numpy as np
import time

from src.config import ModelConfig, CLASS_NAMES, CHUNKING_POLICIES
from src.model import VideoFormatClassifier, GatedTemporalAttention
from src.features import FeatureExtractor
from src.metrics import (
    compute_standard_metrics,
    compute_bootstrap_confidence_intervals,
    compute_peas_score
)

def test_model_initialization_and_size():
    cfg = ModelConfig()
    model = VideoFormatClassifier(cfg)
    param_count = sum(p.numel() for p in model.parameters())
    
    # Assert model is lightweight (< 500,000 parameters / < 2 MB)
    assert param_count < 500_000, f"Model parameter count {param_count} is too high for a lightweight classifier"
    assert param_count > 10_000

def test_gated_attention_forward():
    attn_layer = GatedTemporalAttention(in_dim=128, hidden_dim=64)
    dummy_seq = torch.randn(4, 8, 128) # Batch=4, SeqLen=8, Dim=128
    pooled, weights = attn_layer(dummy_seq)

    assert pooled.shape == (4, 128)
    assert weights.shape == (4, 8)
    # Weights should sum to 1.0 along the temporal dimension
    sums = torch.sum(weights, dim=1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

def test_model_prediction_and_chunking_policy():
    cfg = ModelConfig()
    model = VideoFormatClassifier(cfg)
    dummy_input = torch.randn(1, 8, cfg.visual_dim + cfg.motion_dim)
    
    pred = model.predict(dummy_input)
    assert "category" in pred
    assert pred["category"] in CLASS_NAMES
    assert "confidence" in pred
    assert 0.0 <= pred["confidence"] <= 1.0
    assert "chunking_strategy" in pred
    assert pred["chunking_strategy"] == CHUNKING_POLICIES[pred["category"]]["strategy"]
    assert "chunking_trigger" in pred

def test_cpu_inference_latency():
    cfg = ModelConfig()
    model = VideoFormatClassifier(cfg)
    dummy_input = torch.randn(1, 8, cfg.visual_dim + cfg.motion_dim)

    # Warmup
    for _ in range(5):
        _ = model.predict(dummy_input)

    # Benchmark 20 iterations
    latencies = []
    for _ in range(20):
        t0 = time.perf_counter()
        _ = model.predict(dummy_input)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    p95_latency = np.percentile(latencies, 95)
    print(f"\n[Test Latency] CPU p95: {p95_latency:.2f} ms")
    # Assert latency is ultra-fast (< 50ms on CPU)
    assert p95_latency < 50.0, f"p95 latency {p95_latency}ms exceeds 50ms limit"

def test_peas_metric_and_bootstrap_ci():
    y_true = [0, 1, 2, 0, 1, 2, 3, 4, 5, 0]
    y_pred = [0, 1, 2, 0, 1, 1, 3, 4, 5, 0]
    
    std = compute_standard_metrics(y_true, y_pred)
    assert std["accuracy"] > 0.8
    assert std["macro_f1"] > 0.8

    ci = compute_bootstrap_confidence_intervals(y_true, y_pred, n_bootstraps=100)
    assert "accuracy_ci" in ci
    assert ci["accuracy_ci"][0] <= ci["accuracy_ci"][1]

    peas = compute_peas_score(std["macro_f1"], latency_p95_ms=5.0, param_count=85000)
    assert peas > 0.0
