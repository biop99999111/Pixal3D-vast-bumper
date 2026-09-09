import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

from .common import ROOT, checked_config, digest, object_digest, read_json, source_digest, write_json
from .labels import mask_to_box, overlay, write_labels
from .quality import require_review


def blender_run(blender, job, job_path):
    write_json(job_path, job)
    with Path(job_path).with_suffix(".log").open("w", encoding="utf-8") as log:
        process = subprocess.run([str(blender), "--background", "--python-exit-code", "1",
                                  "--python", str(ROOT / "bumper_synth/blender_worker.py"),
                                  "--", "--job", str(Path(job_path).resolve())],
                                 stdout=log, stderr=subprocess.STDOUT, check=False)
    if process.returncode:
        raise RuntimeError(f"Blender failed; see {Path(job_path).with_suffix('.log')}")


def fixture(blender, output):
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    blender_run(blender, {"fixture": True, "output": str(output)}, output.with_suffix(".job.json"))
    write_json(output.with_suffix(".fixture.json"), {"sha256": digest(output), "not_for_training": True})


def build(asset, config, output, blender, group, split="train", demo=False):
    cfg = checked_config(config, "render")
    if cfg["count"] < 1 or not 0 <= cfg["normal_fraction"] <= 1:
        raise ValueError("Invalid sample count or normal_fraction")
    if not cfg["classes"] or any(c not in (0, 1, 2) for c in cfg["classes"]):
        raise ValueError("classes must contain only 0=dent, 1=ding, 2=scratch")
    if not group.strip() or split not in ("train", "val", "test"):
        raise ValueError("A nonempty source group and valid split are required")
    asset, output = Path(asset).resolve(), Path(output).resolve()
    is_fixture = asset.with_suffix(".fixture.json").exists()
    if demo and not is_fixture:
        raise ValueError("--demo only accepts an asset made by the fixture command")
    if is_fixture and not demo:
        raise ValueError("Fixture datasets require --demo and are not training data")
    if not demo:
        require_review(asset)
    output.mkdir(parents=True, exist_ok=True)
    signature = object_digest({"asset": digest(asset), "config": cfg, "group": group,
                               "split": split, "demo": demo, "code": source_digest()})
    manifest_path = output / "dataset.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if manifest["signature"] != signature:
            raise ValueError("Dataset config/asset/code changed; use a new output directory")
    else:
        manifest = {"signature": signature, "asset_sha256": digest(asset), "group": group,
                    "split": split, "demo_only": demo, "config": cfg, "samples": [], "status": "running"}
    for folder in (f"images/{split}", f"labels/{split}", "masks", "previews", "metadata", "work"):
        (output / folder).mkdir(parents=True, exist_ok=True)
    write_json(manifest_path, manifest)
    completed = {s["index"]: s for s in manifest["samples"]}
    normal_count = round(cfg["count"]*cfg["normal_fraction"])
    for index in range(cfg["count"]):
        previous = completed.get(index)
        if previous and all((output / p).exists() and digest(output / p) == h
                            for p, h in previous["files"].items()):
            print(f"Reuse {index+1}/{cfg['count']}", flush=True)
            continue
        cls = -1 if index < normal_count else cfg["classes"][(index-normal_count) % len(cfg["classes"])]
        sample_id = f"{object_digest(group)[:8]}_{index:06d}"
        for attempt in range(cfg["max_attempts"]):
            work = output / "work" / f"{sample_id}_{attempt}"
            work.mkdir(parents=True, exist_ok=True)
            seed = cfg["seed"] + index*1000 + attempt
            job = {"asset": str(asset), "output": str(work), "seed": seed, "class_id": cls,
                   "config": cfg}
            try:
                blender_run(blender, job, work / "job.json")
            except RuntimeError:
                # Runtime/installation errors must surface, not be hidden by retries.
                manifest.update(status="failed", failed_sample=index, log=str(work / "job.log"))
                write_json(manifest_path, manifest)
                raise
            rows = []
            appearance_delta = None
            mask = np.zeros((cfg["resolution"], cfg["resolution"]), dtype=bool)
            if cls >= 0:
                mask = np.asarray(Image.open(work / "mask.png").convert("L")) > 127
                box = mask_to_box(mask, cfg["min_mask_pixels"], cfg["min_mask_side"])
                if box is None:
                    continue
                rgb = np.asarray(Image.open(work / "rgb.png").convert("RGB"), dtype=float)
                clean = np.asarray(Image.open(work / "clean.png").convert("RGB"), dtype=float)
                appearance_delta = float(np.abs(rgb-clean)[mask].mean())
                if appearance_delta < cfg.get("min_damage_mean_delta", 1.0):
                    continue
                rows = [[cls, *box]]
            break
        else:
            manifest.update(status="failed", failed_sample=index, reason="No sufficiently visible damage")
            write_json(manifest_path, manifest)
            raise ValueError("Damage too small/occluded for labels; inspect work renders and adjust config")
        image_path = output / "images" / split / f"{sample_id}.png"
        label_path = output / "labels" / split / f"{sample_id}.txt"
        mask_path = output / "masks" / f"{sample_id}.png"
        meta_path = output / "metadata" / f"{sample_id}.json"
        preview_path = output / "previews" / f"{sample_id}.png"
        shutil.copyfile(work / "rgb.png", image_path)
        Image.fromarray(mask.astype(np.uint8)*255).save(mask_path)
        write_labels(label_path, rows)
        metadata = read_json(work / "metadata.json")
        metadata.update(group=group, split=split, asset_sha256=digest(asset), demo_only=demo,
                        damage_vs_clean_mae_0_255=appearance_delta)
        write_json(meta_path, metadata)
        overlay(image_path, rows, preview_path)
        record = {"index": index, "id": sample_id, "class_id": cls, "seed": seed,
                  "files": {str(p.relative_to(output)).replace("\\", "/"): digest(p)
                            for p in (image_path, label_path, mask_path, meta_path, preview_path)}}
        manifest["samples"] = [s for s in manifest["samples"] if s["index"] != index] + [record]
        write_json(manifest_path, manifest)
        print(f"Rendered {index+1}/{cfg['count']}: class={cls}", flush=True)
    manifest["status"] = "complete"
    write_json(manifest_path, manifest)
    print(f"Dataset complete: {output}. Inspect previews before training.")


def assemble(datasets, output):
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Assembly output must be empty")
    manifests = [(Path(p).resolve(), read_json(Path(p) / "dataset.json")) for p in datasets]
    groups, assets, names = {}, {}, set()
    for _, manifest in manifests:
        if manifest["status"] != "complete" or manifest["demo_only"]:
            raise ValueError("Only complete non-fixture datasets may be assembled")
        split = manifest["split"]
        for mapping, key in ((groups, manifest["group"]), (assets, manifest["asset_sha256"])):
            if key in mapping and mapping[key] != split:
                raise ValueError("Source group or identical asset leaked across train/val/test")
            mapping[key] = split
        for sample in manifest["samples"]:
            if sample["id"] in names:
                raise ValueError("Duplicate sample IDs; use distinct source groups or combine settings in one batch")
            names.add(sample["id"])
    if not {"train", "val"}.issubset(set(groups.values())):
        raise ValueError("Provide independent source groups for train and val; do not split views of one bumper")
    for source, manifest in manifests:
        for sample in manifest["samples"]:
            for relative, checksum in sample["files"].items():
                if digest(source / relative) != checksum:
                    raise ValueError(f"Changed dataset file: {relative}")
                target = output / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source / relative, target)
    yaml = f"path: {json_quote(str(output))}\ntrain: images/train\nval: images/val\n"
    if "test" in groups.values():
        yaml += "test: images/test\n"
    yaml += "names:\n  0: dent\n  1: ding\n  2: scratch\n"
    (output / "data.yaml").write_text(yaml, encoding="utf-8")
    write_json(output / "sources.json", {"groups": groups, "assets": assets})


def json_quote(value):
    import json
    return json.dumps(value)
