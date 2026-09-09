"""Resolve HF revisions once, download selected inference weights, write a local pipeline.

HF credentials are read by huggingface_hub from HF_TOKEN/login cache, never saved here.
"""
import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bumper_synth.common import ROOT, read_json, safe_error, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="cache/models")
    parser.add_argument("--background-mode", choices=("auto", "provided_mask"), default="provided_mask")
    parser.add_argument("--model", default="TencentARC/Pixal3D")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--refresh", action="store_true", help="Resolve new revisions instead of reusing the lock")
    args = parser.parse_args()
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    api = HfApi()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock_path = output / "models.lock.json"
    previous = read_json(lock_path) if lock_path.exists() and not args.refresh else {}
    revisions = dict(previous.get("revisions", {}))

    def revision(repo, requested="main"):
        if repo not in revisions:
            revisions[repo] = api.model_info(repo, revision=requested).sha
        return revisions[repo]

    def download(repo, patterns=None, requested="main"):
        print(f"Preparing {repo} (credentials are not logged)", flush=True)
        return snapshot_download(repo, revision=revision(repo, requested), allow_patterns=patterns)

    config_file = hf_hub_download(args.model, "pipeline.json", revision=revision(args.model, args.revision))
    config = copy.deepcopy(read_json(config_file))
    selected = {"sparse_structure_flow_model", "sparse_structure_decoder", "shape_slat_flow_model_512",
                "shape_slat_flow_model_1024", "shape_slat_decoder", "tex_slat_flow_model_512",
                "tex_slat_flow_model_1024", "tex_slat_decoder"}
    config["args"]["models"] = {k: v for k, v in config["args"]["models"].items() if k in selected}
    for name, value in config["args"]["models"].items():
        # Upstream first tries this repo, then a fully qualified external repo/path.
        try:
            base = download(args.model, [value+".json", value+".safetensors"], args.revision)
            checkpoint = Path(base) / value
            if not Path(str(checkpoint)+".json").exists() or not Path(str(checkpoint)+".safetensors").exists():
                raise FileNotFoundError(value)
        except FileNotFoundError:
            parts = value.split("/")
            if len(parts) < 3:
                raise ValueError(f"Missing checkpoint {name}: {value}")
            repo, prefix = "/".join(parts[:2]), "/".join(parts[2:])
            base = download(repo, [prefix+".json", prefix+".safetensors"])
            checkpoint = Path(base) / prefix
            if not Path(str(checkpoint)+".safetensors").exists():
                raise FileNotFoundError(f"Missing checkpoint weights: {name}")
        config["args"]["models"][name] = str(checkpoint)
    paths = {
        "dino": download("camenduru/dinov3-vitl16-pretrain-lvd1689m"),
        "moge": str(Path(download("Ruicheng/moge-2-vitl", ["model.pt"])) / "model.pt"),
    }
    if args.background_mode == "auto":
        rembg = config["args"]["rembg_model"]
        repo = rembg.get("args", {}).get("model_name", "ZhengPeng7/BiRefNet")
        paths["rembg"] = download(repo)
        rembg.setdefault("args", {})["model_name"] = paths["rembg"]
    local_pipeline = output / "pipeline"
    write_json(local_pipeline / "pipeline.json", config)
    paths["pixal"] = str(local_pipeline)
    naf_ref = previous.get("naf_ref")
    if not naf_ref:
        naf_ref = read_json(ROOT / "configs/vast/source-refs.json")["naf"]
    # NAF downloads additional weights inside torch.hub; prepare them now on CPU.
    import torch
    torch.hub.load(f"valeoai/NAF:{naf_ref}", "naf", pretrained=True, device="cpu", trust_repo=True)
    lock = {"model": args.model, "revisions": revisions, "paths": paths, "naf_ref": naf_ref,
            "background_mode": args.background_mode,
            "note": "Local absolute paths are machine-specific. Run this script on the rented server."}
    write_json(lock_path, lock)
    print(f"Model preparation complete: {lock_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(json.dumps(safe_error(error)), file=sys.stderr)
        sys.exit(1)
