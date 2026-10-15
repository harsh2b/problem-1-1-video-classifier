"""Gated attention classifier for video format prediction."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional
import numpy as np

from src.config import ModelConfig, CHUNKING_POLICIES, CLASS_NAMES, NUM_CLASSES

class GatedTemporalAttention(nn.Module):
    """Gated attention pooling over temporal frame sequences (Ilse et al., ICML 2018)."""
    def __init__(self, in_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.fc_tanh = nn.Linear(in_dim, hidden_dim)
        self.fc_sigm = nn.Linear(in_dim, hidden_dim)
        self.fc_weight = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [Batch, SeqLen, InDim]
        a_tanh = torch.tanh(self.fc_tanh(x))
        a_sigm = torch.sigmoid(self.fc_sigm(x))
        gated = a_tanh * a_sigm
        raw_weights = self.fc_weight(gated) # [Batch, SeqLen, 1]
        attention_weights = F.softmax(raw_weights, dim=1) # [Batch, SeqLen, 1]
        
        # Weighted sum pooling
        pooled = torch.sum(x * attention_weights, dim=1) # [Batch, InDim]
        return pooled, attention_weights.squeeze(-1)

class VideoFormatClassifier(nn.Module):
    """Classifier head with attention pooling and calibrated softmax."""
    def __init__(self, config: Optional[ModelConfig] = None):
        super().__init__()
        self.config = config or ModelConfig()
        in_dim = self.config.visual_dim + self.config.motion_dim

        # Temporal dimension reduction & projection
        self.projection = nn.Sequential(
            nn.Linear(in_dim, self.config.hidden_dim),
            nn.LayerNorm(self.config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.config.dropout)
        )

        # Gated Temporal Attention
        self.attention_pool = GatedTemporalAttention(self.config.hidden_dim, self.config.hidden_dim // 2)

        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim // 2, NUM_CLASSES)
        )

        # Temperature parameter for probability calibration
        self.temperature = nn.Parameter(torch.ones(1) * self.config.temperature)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [Batch, SeqLen, InDim]
        Returns:
            logits: [Batch, NUM_CLASSES]
            attention_weights: [Batch, SeqLen]
        """
        projected = self.projection(x)
        pooled, attn_weights = self.attention_pool(projected)
        logits = self.classifier(pooled)
        return logits, attn_weights

    def predict(self, x: torch.Tensor) -> Dict[str, Any]:
        """Run inference with calibrated probs and chunking strategy lookup."""
        self.eval()
        with torch.no_grad():
            if x.dim() == 2:
                x = x.unsqueeze(0) # [1, SeqLen, InDim]

            logits, attn_weights = self.forward(x)
            calibrated_logits = logits / torch.clamp(self.temperature, min=0.1)
            probs = F.softmax(calibrated_logits, dim=-1).squeeze(0).cpu().numpy()

            pred_idx = int(np.argmax(probs))
            confidence = float(probs[pred_idx])
            pred_class = CLASS_NAMES[pred_idx]

            # check if prediction is uncertain
            is_uncertain = confidence < self.config.uncertainty_threshold
            margin = confidence - float(np.partition(probs, -2)[-2]) if len(probs) > 1 else confidence

            # look up the chunking strategy
            chunking_info = CHUNKING_POLICIES.get(pred_class, {
                "strategy": "Standard Fixed-Window (10s)",
                "trigger": "Fallback default",
                "rationale": "Default baseline"
            })

            return {
                "category": pred_class,
                "confidence": round(confidence, 4),
                "margin": round(float(margin), 4),
                "is_uncertain": is_uncertain,
                "escalate_tier2": is_uncertain or (margin < 0.15),
                "probabilities": {CLASS_NAMES[i]: round(float(probs[i]), 4) for i in range(NUM_CLASSES)},
                "attention_weights": attn_weights.squeeze(0).cpu().numpy().tolist(),
                "chunking_strategy": chunking_info["strategy"],
                "chunking_trigger": chunking_info["trigger"],
                "chunking_rationale": chunking_info["rationale"]
            }

if __name__ == "__main__":
    cfg = ModelConfig()
    model = VideoFormatClassifier(cfg)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,} ({total_params * 4 / 1024:.1f} KB)")

    dummy_input = torch.randn(2, 8, cfg.visual_dim + cfg.motion_dim)
    logits, attn = model(dummy_input)
    print(f"Logits shape: {logits.shape}, Attention shape: {attn.shape}")
    res = model.predict(dummy_input[0])
    print(f"Prediction result: {res['category']} (conf: {res['confidence']}) -> Strategy: {res['chunking_strategy']}")
