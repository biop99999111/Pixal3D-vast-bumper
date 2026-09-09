"""Sequential one-variable experiments; each inference runs in a fresh process."""
import argparse
import copy
import csv
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bumper_synth.common import ROOT, read_json, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--mask")
    parser.add_argument("--models-lock", required=True)
    parser.add_argument("--config", default="configs/vast/baseline.json")
    parser.add_argument("--output", default="outputs/experiments")
    parser.add_argument("--steps-sweep", action="store_true")
    args = parser.parse_args()
    base = read_json(args.config)
    configs = []
    for low, resolution in ((True,1024),(False,1024),(False,1536)):
        cfg = copy.deepcopy(base)
        cfg.update(low_vram=low, resolution=resolution)
        configs.append((f"{'low' if low else 'standard'}_{resolution}", cfg))
    if args.steps_sweep:
        for stage in ("shape", "texture"):
            for steps in (18,24):
                cfg = copy.deepcopy(base)
                cfg["samplers"][stage]["steps"] = steps
                configs.append((f"{stage}_{steps}_steps", cfg))
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, cfg in configs:
        config = root / f"{name}.json"
        write_json(config, cfg)
        run = root / name
        command = [sys.executable, "-m", "bumper_synth", "infer", "--config", str(config),
                   "--image", str(Path(args.image).resolve()), "--output", str(run),
                   "--models-lock", str(Path(args.models_lock).resolve())]
        if args.mask:
            command += ["--mask", str(Path(args.mask).resolve())]
        # Continue to the next independent configuration after OOM; never relabel it as success.
        process = subprocess.run(command, cwd=ROOT, check=False)
        stages = read_json(run / "stages.json") if (run / "stages.json").exists() else []
        rows.append({"experiment": name, "exit_code": process.returncode,
                     "peak_reserved_bytes": max((s.get("peak_reserved_bytes",0) for s in stages), default=0),
                     "measured_stage_seconds": sum(s.get("seconds",0) for s in stages)})
        with (root / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
