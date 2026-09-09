from pathlib import Path

import numpy as np
import trimesh

from .common import digest, read_json, write_json


def inspect_glb(path, output):
    scene = trimesh.load(path, force="scene", process=False)
    rows = []
    texture_dir = Path(output).parent / "textures"
    texture_dir.mkdir(parents=True, exist_ok=True)
    for index, (name, geometry) in enumerate(scene.geometry.items()):
        finite = bool(np.isfinite(geometry.vertices).all())
        visual = geometry.visual
        material = getattr(visual, "material", None)
        textures = []
        for channel in ("baseColorTexture", "metallicRoughnessTexture", "normalTexture", "image"):
            image = getattr(material, channel, None)
            if image is not None and hasattr(image, "save"):
                file = texture_dir / f"{index}_{channel}.png"
                image.save(file)
                textures.append({"channel": channel, "size": list(image.size), "file": file.name})
        rows.append({"name": name, "vertices": len(geometry.vertices), "faces": len(geometry.faces),
                     "finite": finite, "watertight": bool(geometry.is_watertight),
                     "uv": getattr(visual, "uv", None) is not None, "textures": textures})
    result = {"sha256": digest(path), "geometry": rows,
              "structural_pass": bool(rows) and all(r["finite"] and r["faces"] > 0 for r in rows),
              "semantic_status": "needs_human_review",
              "note": "Trimesh load check, not Khronos validation or proof of bumper-only geometry."}
    write_json(output, result)
    return result


def approve_asset(asset, note):
    if not note.strip():
        raise ValueError("Review note required: bumper-only geometry, clean surface, valid textures")
    path = Path(asset).with_suffix(".review.json")
    write_json(path, {"sha256": digest(asset), "accepted": True, "note": note})
    return path


def require_review(asset):
    path = Path(asset).with_suffix(".review.json")
    if not path.exists():
        raise ValueError("Review the GLB, then run: python -m bumper_synth review --asset ... --note ...")
    value = read_json(path)
    if value.get("accepted") is not True or value.get("sha256") != digest(asset):
        raise ValueError("Review missing or asset changed since review")
