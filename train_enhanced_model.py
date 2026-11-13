"""Train enhanced video format classifier and update model weights."""
import time
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

from src.model import VideoFormatClassifier
from src.config import ModelConfig, CLASS_NAMES
from src.metrics import (
    compute_standard_metrics,
    compute_bootstrap_confidence_intervals,
    compute_peas_score,
    LatencyProfiler
)

def main():
    data_dir = Path("data")
    cache_path = data_dir / "features_cache.npz"
    weights_path = data_dir / "model_weights.pt"

    cache = np.load(cache_path)
    X_tr, y_tr = cache["X_train"], cache["y_train"]
    X_te, y_te = cache["X_test"], cache["y_test"]

    cfg = ModelConfig()
    torch.manual_seed(7)
    np.random.seed(7)

    model = VideoFormatClassifier(cfg)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.08)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.8e-3, weight_decay=8e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=45)

    tr_ds = torch.utils.data.TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
    loader = torch.utils.data.DataLoader(tr_ds, batch_size=16, shuffle=True)

    print("Training Residual Gated Attention Classifier (Seed 7)...")
    for epoch in range(45):
        model.train()
        for bx, by in loader:
            optimizer.zero_grad()
            out, _ = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
        scheduler.step()

    torch.save(model.state_dict(), weights_path)
    print(f"Saved enhanced weights to {weights_path} ({weights_path.stat().st_size / 1024:.1f} KB)")

    # Benchmark
    model.eval()
    profiler = LatencyProfiler()
    y_preds = []

    with torch.no_grad():
        for i in range(len(X_te)):
            sample_t = torch.from_numpy(X_te[i]).unsqueeze(0)
            t0 = time.perf_counter()
            pred = model.predict(sample_t)
            profiler.record(time.perf_counter() - t0)
            y_preds.append(CLASS_NAMES.index(pred["category"]))

    std = compute_standard_metrics(y_te.tolist(), y_preds)
    ci = compute_bootstrap_confidence_intervals(y_te.tolist(), y_preds)
    lat = profiler.summary()
    param_count = sum(p.numel() for p in model.parameters())
    peas = compute_peas_score(std["macro_f1"], lat["p95_ms"], param_count)

    print("\n================ BENCHMARK REPORT (ENHANCED MODEL) ================")
    print(f"Accuracy:          {std['accuracy'] * 100:.2f}% ({sum(np.array(y_preds) == y_te)}/23 correct)")
    print(f"Macro-F1:          {std['macro_f1']:.4f} (95% CI: {ci['macro_f1_ci']})")
    print(f"Macro-Precision:   {std['macro_precision']:.4f}")
    print(f"Macro-Recall:      {std['macro_recall']:.4f}")
    print(f"Latency (p50/p95): {lat['p50_ms']} ms / {lat['p95_ms']} ms")
    print(f"Throughput:        {lat['throughput_fps']} inf/sec")
    print(f"Model Parameters:  {param_count:,} ({round(param_count*4/1024, 2)} KB)")
    print(f"PEAS Score:        {peas}")
    print(f"Per-Class F1:      {std['per_class_f1']}")
    print("===================================================================")

if __name__ == "__main__":
    main()
