"""Feature extraction: visual embeddings + motion dynamics from video frames."""
import os
import cv2
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path
from typing import List, Tuple, Dict, Optional

from src.config import ModelConfig

class FeatureExtractor:
    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig()
        self.device = torch.device(self.config.device)

        # MobileNetV3-Small for visual features (~9MB, 576-d output)
        try:
            weights = models.MobileNet_V3_Small_Weights.DEFAULT
            base_model = models.mobilenet_v3_small(weights=weights)
        except Exception as e:
            print(f"Warning: remote weights unavailable ({e}), using random init")
            base_model = models.mobilenet_v3_small(weights=None)
        self.visual_backbone = base_model.features.to(self.device).eval()
        self.pooling = base_model.avgpool.to(self.device)

        # Vision preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((self.config.frame_size, self.config.frame_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def sample_keyframes(self, video_path: str, num_frames: Optional[int] = None) -> Tuple[List[np.ndarray], List[float]]:
        """Sample N keyframes uniformly across the video timeline."""
        n_frames = num_frames or self.config.num_frames
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            # Return synthetic frames if video cannot be opened
            return [np.zeros((self.config.frame_size, self.config.frame_size, 3), dtype=np.uint8) for _ in range(n_frames)], [i / n_frames for i in range(n_frames)]

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            total_frames = 100

        # evenly spaced across the video
        frame_indices = np.linspace(0, max(0, total_frames - 1), n_frames, dtype=int)
        sampled_frames = []
        timestamps = []

        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if ret and frame is not None:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                sampled_frames.append(frame_rgb)
            else:
                sampled_frames.append(np.zeros((self.config.frame_size, self.config.frame_size, 3), dtype=np.uint8))
            timestamps.append(float(idx) / float(total_frames))

        cap.release()
        return sampled_frames, timestamps

    def compute_motion_dynamics(self, frames: List[np.ndarray]) -> np.ndarray:
        """Compute motion deltas between consecutive frames. Returns [N, 8] features."""
        n = len(frames)
        motion_features = []
        grays = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames]

        for i in range(n):
            if i == 0:
                diff = np.zeros_like(grays[0])
            else:
                diff = cv2.absdiff(grays[i], grays[i - 1])

            # basic stats from the frame diff
            mean_motion = np.mean(diff) / 255.0
            std_motion = np.std(diff) / 255.0
            p90_motion = np.percentile(diff, 90) / 255.0

            # edge density via sobel
            sobelx = cv2.Sobel(grays[i], cv2.CV_32F, 1, 0, ksize=3)
            sobely = cv2.Sobel(grays[i], cv2.CV_32F, 0, 1, ksize=3)
            edge_energy = np.mean(np.sqrt(sobelx**2 + sobely**2)) / 255.0

            feat = [
                mean_motion,
                std_motion,
                p90_motion,
                edge_energy,
                float(mean_motion > 0.15),
                float(mean_motion > 0.30),
                np.max(diff) / 255.0,
                float(i) / float(n) # Normalized time
            ]
            motion_features.append(feat)

        return np.array(motion_features, dtype=np.float32)

    @torch.no_grad()
    def extract_visual_features(self, frames: List[np.ndarray]) -> np.ndarray:
        """Run MobileNetV3 on frames, returns [N, 576] feature array."""
        tensors = []
        for f in frames:
            pil_img = Image.fromarray(f)
            tensors.append(self.transform(pil_img))
        batch = torch.stack(tensors).to(self.device)
        feats = self.visual_backbone(batch)
        pooled = self.pooling(feats).flatten(1)
        return pooled.cpu().numpy()

    def extract_video_features(self, video_path: str) -> Dict[str, np.ndarray]:
        """Full feature pipeline for one video. Returns visual, motion, and combined tensors."""
        frames, timestamps = self.sample_keyframes(video_path)
        visual_feats = self.extract_visual_features(frames)
        motion_feats = self.compute_motion_dynamics(frames)

        # Concatenate visual + motion into temporal sequence
        combined = np.concatenate([visual_feats, motion_feats], axis=-1)

        return {
            "visual": visual_feats,
            "motion": motion_feats,
            "temporal_seq": combined,
            "timestamps": np.array(timestamps, dtype=np.float32)
        }

if __name__ == "__main__":
    extractor = FeatureExtractor()
    print("Feature extractor ready.")
    # Test on synthetic frame array
    dummy_frames = [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(8)]
    vis = extractor.extract_visual_features(dummy_frames)
    mot = extractor.compute_motion_dynamics(dummy_frames)
    print(f"Visual shape: {vis.shape}, Motion shape: {mot.shape}")
