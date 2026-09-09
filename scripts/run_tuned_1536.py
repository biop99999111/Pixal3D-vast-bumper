"""Repeat the saved bumper run with sparse projection and strict 1536 resolution."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bumper_synth.common import checked_config, digest, read_json, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous", default="outputs/bumper_single_001/run.json")
    parser.add_argument("--image", default="assets/images/bumper.jpg")
    parser.add_argument("--mask")
    parser.add_argument("--models-lock", default="cache/models/models.lock.json")
    parser.add_argument("--output", default="outputs/bumper_1536_sparse_001")
    parser.add_argument("--chunk-size", type=int, default=8192)
    parser.add_argument("--bumper-only", action="store_true",
                        help="Experimental silhouette + front-depth support constraint")
    parser.add_argument("--depth-min", type=float, default=0.15)
    parser.add_argument("--depth-max", type=float, default=0.5)
    args = parser.parse_args()
    previous = read_json(args.previous)
    if digest(args.image) != previous["input_sha256"]:
        raise ValueError("Image differs from the previous run")
    read_json(args.models_lock)
    cfg = dict(previous["config"])
    cfg.update(resolution=1536, low_vram=False,
               projection_chunk_size=args.chunk_size, require_requested_resolution=True)
    reusable_mask = None
    if args.bumper_only:
        cfg["background_mode"] = "provided_mask"
        cfg["bumper_constraint"] = {"depth_range": [args.depth_min, args.depth_max],
                                    "silhouette_margin_pixels": 3}
        if not args.mask:
            # Only the inspected, isolated example bumper is eligible for automatic reuse.
            if previous["input_sha256"] != "34832200ed35093270c59674e2bf4ca690dd4f07515e890136ec8c8b0ef7f8a5":
                raise ValueError("For another image, supply a bumper-only mask with --mask")
            from PIL import Image
            foreground = Image.open(Path(args.previous).parent / "foreground_rgba.png")
            if foreground.mode != "RGBA":
                raise ValueError("Previous foreground must have an alpha channel")
            reusable_mask = foreground.getchannel("A")
            if reusable_mask.size != Image.open(args.image).size:
                raise ValueError("Previous mask dimensions differ; supply an original-size --mask")
    if cfg["background_mode"] == "provided_mask" and not args.mask and reusable_mask is None:
        raise ValueError("Previous run requires --mask")
    if args.chunk_size < 1:
        raise ValueError("--chunk-size must be positive")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    if reusable_mask is not None:
        args.mask = str(output / "bumper_mask_original.png")
        reusable_mask.save(args.mask)
    config_path = output / "config.json"
    write_json(config_path, cfg)
    checked_config(config_path, "inference")
    if args.bumper_only:
        from bumper_synth.preprocess import prepare
        prepare(args.image, output / "input_preview", mode="provided_mask",
                mask_path=args.mask, padding=cfg["padding"])
        print(f"[Bumper input] {output / 'input_preview/model_input.png'}", flush=True)
    command = [sys.executable, "-u", "-m", "bumper_synth", "infer",
               "--config", str(config_path), "--image", str(Path(args.image).resolve()),
               "--models-lock", str(Path(args.models_lock).resolve()), "--output", str(output)]
    if args.mask:
        command.extend(["--mask", str(Path(args.mask).resolve())])
    print(json.dumps({"resolution": 1536, "low_vram": False,
                      "chunk_size": args.chunk_size, "output": str(output)}, indent=2), flush=True)
    monitor = None
    with (output / "gpu-memory.csv").open("w", encoding="utf-8") as gpu_log:
        try:
            monitor = subprocess.Popen([
                "nvidia-smi", "--query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu",
                "--format=csv", "-lms", "500"], stdout=gpu_log, stderr=subprocess.DEVNULL)
            with (output / "console.log").open("w", encoding="utf-8") as log:
                process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                                           stderr=subprocess.STDOUT, text=True, errors="replace")
                try:
                    for line in process.stdout:
                        print(line, end="", flush=True)
                        log.write(line)
                        log.flush()
                    return process.wait()
                finally:
                    if process.poll() is None:
                        process.terminate()
                        process.wait()
        finally:
            if monitor is not None:
                monitor.terminate()
                monitor.wait()


if __name__ == "__main__":
    sys.exit(main())
