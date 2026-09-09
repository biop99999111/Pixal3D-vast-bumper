"""Experimental front-view inpainting + projection bake; geometry stays unchanged.

Always produces a NEW unreviewed asset. Never postprocesses labelled damage data.
"""
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from .alignment import camera_to_blender
from .common import digest, read_json, write_json
from .dataset import blender_run


def run(run_dir, output, model, prompt, blender, seed=42, strength=.25, revision="main"):
    if not 0 < strength <= .5:
        raise ValueError("Experimental texture strength must be > 0 and <= 0.5")
    run_dir, output = Path(run_dir).resolve(), Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use an empty retexture output directory")
    source = run_dir / "alignment/base_color.png"
    if not source.exists():
        raise ValueError("Run align first; retexture operates on the clean reference view only")
    output.mkdir(parents=True, exist_ok=True)
    import torch
    from diffusers import AutoPipelineForInpainting
    from huggingface_hub import HfApi, snapshot_download
    resolved = HfApi().model_info(model, revision=revision).sha
    local = snapshot_download(model, revision=resolved)
    pipe = AutoPipelineForInpainting.from_pretrained(local, torch_dtype=torch.float16)
    pipe.to("cuda")
    image = Image.open(source).convert("RGB")
    silhouette = Image.open(run_dir / "alignment/silhouette.png").convert("L")
    # Keep a narrow border of the original silhouette to reduce edge drift.
    mask = silhouette.filter(ImageFilter.MinFilter(7))
    size = (max(64,image.width//8*8),max(64,image.height//8*8))
    result = pipe(prompt=prompt, image=image.resize(size), mask_image=mask.resize(size),
                  width=size[0], height=size[1], strength=strength, num_inference_steps=40,
                  generator=torch.Generator("cuda").manual_seed(seed)).images[0]
    result = result.resize(image.size, Image.Resampling.LANCZOS)
    Image.composite(result, image, mask).save(output / "refined_front.png")
    del pipe
    import gc
    gc.collect(); torch.cuda.empty_cache()
    camera = read_json(run_dir / "camera.json")
    job = {"retexture": True, "asset": str(run_dir / "bumper.glb"),
           "image": str(output / "refined_front.png"), "output": str(output),
           "camera_matrix": camera_to_blender(camera).tolist(), "fov": camera["camera_angle_x"],
           "texture_size": 2048}
    blender_run(blender, job, output / "bake.job.json")
    write_json(output / "camera.json", camera)
    write_json(output / "retexture.json", {"source_sha256": digest(run_dir / "bumper.glb"),
               "model": model, "revision": resolved, "prompt": prompt, "seed": seed,
               "strength": strength, "status": "needs_human_review",
               "scope": "Front-view Base Color only; hidden surfaces retain original color. Review seams, lighting baked into color and invented damage."})
