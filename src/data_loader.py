"""
Data Loader and Dataset Management for Task 1: Video Classifier.
Ingests Video-MME 10% sample metadata, manages video downloads from HuggingFace,
and prepares stratified train/test partitions by video_id to avoid data leakage.
"""
import os
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.config import SUB_TO_CATEGORY, CLASS_NAMES

REPO_ID = "aintropy-ai/video-mme-sample10"
CACHE_DIR = Path("data")

class VideoMMELoader:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.videos_dir = self.data_dir / "videos"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.videos_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.data_dir / "video_metadata.csv"

    def load_metadata(self) -> pd.DataFrame:
        """
        Loads metadata from HuggingFace and consolidates by unique video_id.
        Maps sub_category to the 6 Chunking Categories.
        """
        if self.metadata_file.exists():
            return pd.read_csv(self.metadata_file)

        print("[DataLoader] Fetching Video-MME 10% sample from Hugging Face...")
        ds = load_dataset(REPO_ID, split="test")
        df = ds.to_pandas()

        # Consolidate per video_id (each video has 3 QA pairs)
        unique_videos = df.drop_duplicates(subset=["video_id"]).copy()
        
        # Map sub_category to chunking category
        unique_videos["chunk_category"] = unique_videos["sub_category"].map(
            lambda x: SUB_TO_CATEGORY.get(x, "Performance_Narrative")
        )
        unique_videos["class_idx"] = unique_videos["chunk_category"].map(
            lambda c: CLASS_NAMES.index(c)
        )

        unique_videos.to_csv(self.metadata_file, index=False)
        print(f"[DataLoader] Processed {len(unique_videos)} unique videos. Saved to {self.metadata_file}")
        return unique_videos

    def download_video(self, video_id: str) -> Optional[str]:
        """
        Downloads a specific video MP4 from HuggingFace repository if not already cached.
        """
        local_path = self.videos_dir / f"{video_id}.mp4"
        if local_path.exists() and local_path.stat().st_size > 0:
            return str(local_path)

        try:
            filename = f"videos/{video_id}.mp4"
            downloaded = hf_hub_download(
                repo_id=REPO_ID,
                filename=filename,
                repo_type="dataset",
                local_dir=str(self.data_dir),
                local_dir_use_symlinks=False
            )
            return downloaded
        except Exception as e:
            print(f"[DataLoader] Warning: Could not download {video_id}: {e}")
            return None

    def get_splits(
        self, 
        test_size: float = 0.25, 
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Returns stratified train and test dataframes split strictly by video_id.
        """
        df = self.load_metadata()
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            stratify=df["chunk_category"],
            random_state=random_state
        )
        return train_df.reset_index(drop=True), test_df.reset_index(drop=True)

if __name__ == "__main__":
    loader = VideoMMELoader()
    df = loader.load_metadata()
    print("\n[DataLoader] Class distribution across 90 videos:")
    print(df["chunk_category"].value_counts())
