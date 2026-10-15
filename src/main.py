"""
Main CLI Entrypoint for Task 1: Video Classifier.
Invoked by AIntropy Grader Bot via run.yaml.
Supports:
    --benchmark : Run full evaluation and compute PEAS score with 95% Bootstrap CI.
    --train     : Train or retrain model on Video-MME dataset.
    --predict   : Run end-to-end inference on a video file.
    --json      : Output results as machine-readable JSON.
"""
import argparse
import json
import sys
from pathlib import Path

# Add root directory to sys.path so 'src' is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.trainer import ClassifierPipeline
from src.config import ModelConfig

def main():
    parser = argparse.ArgumentParser(
        description="AIntropy Task 1: Lightweight Multimodal Video Classifier for Chunking Strategy Optimization."
    )
    parser.add_argument(
        "--benchmark", 
        action="store_true", 
        help="Run full evaluation benchmark on Video-MME test split."
    )
    parser.add_argument(
        "--train", 
        action="store_true", 
        help="Train the Gated Attention classifier on the dataset."
    )
    parser.add_argument(
        "--predict", 
        type=str, 
        help="Path to an MP4 video file or video_id to classify."
    )
    parser.add_argument(
        "--json", 
        action="store_true", 
        help="Format output as JSON."
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=25, 
        help="Number of training epochs."
    )

    args = parser.parse_args()
    config = ModelConfig(epochs=args.epochs)
    pipeline = ClassifierPipeline(config=config)

    if args.train:
        print("[Task 1] Initiating training run...")
        res = pipeline.train()
        if args.json:
            print(json.dumps(res, indent=2))
        else:
            print("\n>>> Training and Evaluation Completed Successfully <<<")
            print(f"Accuracy: {res['metrics']['accuracy'] * 100:.2f}%")
            print(f"Macro-F1: {res['metrics']['macro_f1']:.4f}")
            print(f"PEAS Score: {res['peas_score']}")

    elif args.predict:
        video_path = args.predict
        if not Path(video_path).exists():
            # Check if it's a video_id in the cache or downloads
            cached = pipeline.loader.download_video(video_path)
            if cached:
                video_path = cached
            else:
                print(f"Error: Video file '{args.predict}' not found.", file=sys.stderr)
                sys.exit(1)

        result = pipeline.predict_video(video_path)
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print("\n================ VIDEO CLASSIFICATION RESULT ================")
            print(f"Target Video:             {video_path}")
            print(f"Predicted Category:       {result['category']}")
            print(f"Confidence:               {result['confidence'] * 100:.2f}%")
            print(f"Uncertainty Flag:         {'YES (Tier-2 Escalation Needed)' if result['is_uncertain'] else 'NO (High Confidence)'}")
            print(f"Recommended Chunking:     {result['chunking_strategy']}")
            print(f"Chunking Trigger Rule:    {result['chunking_trigger']}")
            print(f"Strategic Rationale:      {result['chunking_rationale']}")
            print(f"Inference Latency:        {result['inference_latency_ms']} ms")
            print("==============================================================")

    else:
        # Default action: run benchmark
        print("[Task 1] Running evaluation benchmark on Video-MME 10% sample...")
        res = pipeline.evaluate()
        if args.json:
            print(json.dumps(res, indent=2))
        else:
            print("\n================ AINTROPY EVALUATION REPORT ================")
            print(f"Top-1 Accuracy:           {res['metrics']['accuracy'] * 100:.2f}%")
            print(f"Macro-F1 Score:           {res['metrics']['macro_f1']:.4f}")
            print(f"95% Bootstrap CI:         Accuracy {res['confidence_intervals']['accuracy_ci']}, F1 {res['confidence_intervals']['macro_f1_ci']}")
            print(f"Latency p50 / p95:        {res['latency']['p50_ms']} ms / {res['latency']['p95_ms']} ms")
            print(f"Throughput:               {res['latency']['throughput_fps']} inferences/sec")
            print(f"Model Parameters:         {res['model_parameters']:,} ({res['model_size_kb']} KB)")
            print(f"PEAS Score (Novel):       {res['peas_score']}")
            print("============================================================")

if __name__ == "__main__":
    main()
