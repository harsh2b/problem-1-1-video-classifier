# Task 1: Video Classifier — End-to-End Implementation Plan

> **Problem**: Classify incoming videos across diverse categories (gameplay, sports, podcast, wildlife, egocentric, etc.) to determine the optimal chunking strategy for building a video index that perceives reality.
> **Constraint**: Lightweight, fast at scale, Pareto frontier (Accuracy vs. Latency vs. Model Size), statistical metric rigor.

---

## Architecture Blueprint: Sparse Multimodal Frame-Audio Attention Classifier

```
                   Input Video (MP4 / WebM / Stream)
                                 │
                 ┌───────────────┴───────────────┐
                 ▼                               ▼
       [Visual Sparse Sampler]          [Audio Energy / VAD]
       - 4 to 8 Keyframes               - Speech presence ratio
       - Inter-frame temporal delta     - Audio RMS / Spectral flux
                 │                               │
                 ▼                               ▼
       [Compact Vision Backbone]        [Acoustic Feature Extractor]
       - MobileNetV3-Small / CLIP       - Speech vs. Music vs. Ambience
       - (1024-dim visual vectors)      - (64-dim acoustic vector)
                 │                               │
                 └───────────────┬───────────────┘
                                 ▼
                 [Temporal Feature Concatenation]
                                 │
                                 ▼
             [Gated Attention MIL Pooling (PyTorch)]
             - Learns temporal chunk salience
             - Aggregates sequence into single video vector
                                 │
                                 ▼
                   [Calibrated Classification Head]
                   - Softmax with temperature scaling
                   - Confidence & uncertainty margin check
                                 │
                 ┌───────────────┴───────────────┐
                 ▼                               ▼
     Predicted Video Category       Recommended Chunking Policy
     (e.g., Sports Competition)     (e.g., Dense Optical Flow 2-5s)
```

---

## Chunking Policy Mapping

| Video Category | Visual / Motion Characteristics | Acoustic Signal | Optimal Chunking Policy | Rationale for Video Indexing |
| :--- | :--- | :--- | :--- | :--- |
| **Sports Competition** | High temporal dynamics, camera pans | Cheering, commentary, whistles | **Dense Optical Flow Window (2–5s)** | Captures rapid physical maneuvers without skipping key actions. |
| **Gameplay / Esports** | Persistent HUD, abrupt screen cuts | Game SFX, fast commentary | **UI/State & Event Chunking** | Splits on round resets, inventory opens, health drops. |
| **Podcast / News** | Static talking heads, minimal motion | Continuous alternating speech | **Audio/Transcript Topic Chunking (30–120s)** | Aligns chunks with semantic dialogue turns. |
| **Wildlife / Nature** | Slow pans, zooming, static frames | Ambient background, low speech | **Visual Continuity / Scene Cut Chunking** | Chunks based on subject entrance/exit and habitat changes. |
| **Life Record / Egocentric** | First-person jitter, object manipulation | Daily ambience, intermittent speech | **Hand-Object Interaction Window (5–15s)** | Centers chunks around user physical interactions. |
| **Performance / Stage** | Fixed stage lighting, actors moving | Music, applause, singing | **Acoustic & Lighting Shift Chunking** | Marks musical movements and scene transitions. |

---

## Phase-Wise Execution Steps

### Phase 1: Environment & Dataset Preprocessing
- [x] Dedicated `.venv` with PyTorch, TorchVision, OpenCV, Scikit-learn, Datasets, Jupyter.
- [x] Inspected `aintropy-ai/video-mme-sample10` (90 videos across 6 domains).
- [ ] Implement `src/data_loader.py` to ingest dataset records, metadata, and video streams.

### Phase 2: Feature Engineering (`src/features.py`)
- [ ] **Sparse Frame Extractor**: Uniform keyframe sampling (4–8 frames) with scene-change awareness.
- [ ] **Motion Dynamics**: Inter-frame temporal pixel differences (Optical Flow proxy) to capture kinetic energy in $<2$ms.
- [ ] **Semantic Visual Features**: Pretrained MobileNetV3-Small / CLIP forward pass.
- [ ] **Audio Presence & Energy**: Detect speech vs. ambient noise ratio.

### Phase 3: Model Architecture (`src/model.py`)
- [ ] **Gated Temporal Attention (MIL)**: Learn dynamic attention weights $\alpha_t$ over sampled timestamps.
- [ ] **Classification Head**: Calibrated MLP with temperature scaling.
- [ ] **Fallback / Escalation Trigger**: If $\max(P) < 0.60$, flag as uncertain or escalate to denser sampling.

### Phase 4: Evaluation & Novel Statistical Metric (`src/metrics.py`)
- [ ] Standard Metrics: Top-1 Accuracy, Macro-F1, Confusion Matrix, Latency (p50, p95).
- [ ] **Novel Statistical Metric — PEAS (Pareto Efficiency-Accuracy Score)**:
  $$\text{PEAS} = \frac{\text{Macro-F1}^2}{\log_{10}(\text{Latency}_{\text{p95}} \text{ in ms}) \times \log_2(\text{Params in M} + 1)}$$
- [ ] **Bootstrap 95% Confidence Intervals**: 1,000 bootstrap resamples on test split to report statistically sound error bounds.

### Phase 5: Grader Bot Verification & Test Suite
- [ ] `src/main.py`: CLI supporting `--input <video_path>`, `--benchmark`, and `--output <json>`.
- [ ] `tests/test_classifier.py`: Pytest suite verifying determinism, input shapes, and latency constraints.
- [ ] `run.yaml`: Verified non-interactive setup and execution commands.

### Phase 6: Jupyter Storytelling Notebook & Documentation
- [ ] `notebooks/walkthrough.ipynb`: End-to-end story: dataset exploration, feature extraction visualization, attention weight heatmaps, and Pareto frontier curves.
- [ ] `docs/README.md`: 2-page equivalent technical report including the draw.io architecture diagram and paper deployment architecture (FastAPI + Triton + ONNX Runtime).
