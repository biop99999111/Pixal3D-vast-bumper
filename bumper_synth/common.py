from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def object_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def safe_error(error):
    message = str(error)
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        token = os.getenv(key)
        if token:
            message = message.replace(token, "[REDACTED]")
    return {"type": type(error).__name__, "message": message}


def source_digest():
    files = list((ROOT / "bumper_synth").glob("*.py"))
    files += [ROOT / "inference.py", *list((ROOT / "pixal3d").rglob("*.py"))]
    return object_digest({str(p.relative_to(ROOT)): digest(p) for p in sorted(files)})


def checked_config(path, kind):
    cfg = read_json(path)
    if cfg.get("kind") != kind:
        raise ValueError(f"Expected config kind={kind!r}")
    if kind == "inference":
        if cfg["resolution"] not in (1024, 1536):
            raise ValueError("resolution must be 1024 or 1536")
        if cfg["background_mode"] not in ("auto", "provided_mask"):
            raise ValueError("background_mode must be auto or provided_mask")
        if not isinstance(cfg["low_vram"], bool):
            raise ValueError("low_vram must be a boolean")
        chunk = cfg.get("projection_chunk_size", 8192)
        if type(chunk) is not int or chunk < 1:
            raise ValueError("projection_chunk_size must be a positive integer")
        if not isinstance(cfg.get("require_requested_resolution", False), bool):
            raise ValueError("require_requested_resolution must be a boolean")
        if cfg.get("bumper_constraint"):
            constraint = cfg["bumper_constraint"]
            if cfg["background_mode"] != "provided_mask":
                raise ValueError("bumper_constraint requires provided_mask mode")
            depth = constraint.get("depth_range", [])
            if len(depth) != 2 or not -.5 <= depth[0] < depth[1] <= .5:
                raise ValueError("depth_range must be increasing and within [-0.5, 0.5]")
            margin = constraint.get("silhouette_margin_pixels", 3)
            if type(margin) is not int or not 0 <= margin <= 32:
                raise ValueError("silhouette_margin_pixels must be an integer from 0 to 32")
        for stage in ("ss", "shape", "texture"):
            if not 1 <= cfg["samplers"][stage]["steps"] <= 200:
                raise ValueError("steps must be between 1 and 200")
    return cfg
