from pathlib import Path

import numpy as np
from PIL import Image

from .common import read_json, write_json


def camera_to_blender(camera):
    # Source ProjGrid rotates voxel coordinates by Rx(90); exporter applies its
    # own matrix; Blender imports glTF Y-up with Rx(90). Account for all three.
    grid = np.array([[1,0,0,0],[0,0,-1,0],[0,1,0,0],[0,0,0,1]], dtype=float)
    source_camera = grid.copy()
    source_camera[1,3] = -camera["distance"]
    source_camera[:3,3] *= camera.get("mesh_scale", 1.0)
    exported = np.asarray(camera["export_transform"])
    return grid @ exported @ np.linalg.inv(grid) @ source_camera


def compare(run_dir, blender):
    from .dataset import blender_run
    run = Path(run_dir).resolve()
    output = run / "alignment"
    output.mkdir(parents=True, exist_ok=True)
    camera = read_json(run / "camera.json")
    reference = Image.open(run / "model_input.png").convert("RGB")
    job = {"reference": True, "asset": str(run / "bumper.glb"), "output": str(output),
           "camera_matrix": camera_to_blender(camera).tolist(), "fov": camera["camera_angle_x"],
           "resolution": reference.width}
    blender_run(blender, job, output / "job.json")
    image = Image.open(output / "base_color.png").convert("RGB")
    mask = np.asarray(Image.open(output / "silhouette.png").convert("L")) > 127
    original_mask = np.asarray(Image.open(run / "input_mask.png").convert("L")) > 204
    union = mask | original_mask
    common = mask & original_mask
    a, b = np.asarray(reference).astype(float), np.asarray(image).astype(float)
    Image.blend(reference, image, .5).save(output / "overlay.png")
    Image.fromarray(np.abs(a-b).clip(0,255).astype(np.uint8)).save(output / "difference.png")
    write_json(output / "metrics.json", {
        "silhouette_iou": float(common.sum()/union.sum()) if union.any() else 0,
        "overlap_rgb_mae_0_255": float(np.abs(a-b)[common].mean()) if common.any() else None,
        "note": "Base Color vs photograph includes lighting differences; not a semantic bumper-only score.",
    })
