"""Optional training entry point. Install Ultralytics in a separate environment."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", required=True, help="Chosen YOLO detection checkpoint or YAML")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=768)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--project", default="outputs/yolo")
    parser.add_argument("--real-eval-data", help="Optional independent REAL dataset YAML")
    args = parser.parse_args()
    from ultralytics import YOLO
    model = YOLO(args.model)
    model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                device=args.device, seed=args.seed, project=args.project, name="train")
    if args.real_eval_data:
        result = model.val(data=args.real_eval_data, split="test", imgsz=args.imgsz,
                           device=args.device, project=args.project, name="real_test")
        Path(args.project).mkdir(parents=True, exist_ok=True)
        Path(args.project, "real_test_metrics.json").write_text(json.dumps(result.results_dict, indent=2))


if __name__ == "__main__":
    main()
