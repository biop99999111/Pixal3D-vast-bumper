"""CPU preprocessing equivalent to upstream framing, with preserved masks."""
from pathlib import Path

import numpy as np
from PIL import Image

from .common import write_json


def prepare(image_path, output_dir, mode="auto", mask_path=None, remover=None, padding=1.1):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image = Image.open(image_path)
    original_size = image.size
    if padding < 1:
        raise ValueError("padding must be >= 1")
    if mask_path:
        mask = Image.open(mask_path).convert("L")
        if mask.size != image.size:
            raise ValueError("Mask dimensions must equal original image dimensions")
        image = image.convert("RGBA")
        image.putalpha(mask)
    valid_alpha = image.mode == "RGBA" and np.any(np.asarray(image)[:, :, 3] < 255)
    if mode == "provided_mask" and not valid_alpha:
        raise ValueError("provided_mask requires a non-opaque RGBA image or external mask")
    scale = min(1.0, 1024 / max(image.size))
    resized = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
    image = image.resize(resized, Image.Resampling.LANCZOS)
    if not valid_alpha:
        if remover is None:
            raise ValueError("RGB auto mode requires a background removal model")
        image = remover(image.convert("RGB"))
    image = image.convert("RGBA")
    mask = np.asarray(image)[:, :, 3]
    yy, xx = np.where(mask > 204)
    if len(xx) < 16 or np.ptp(xx) < 2 or np.ptp(yy) < 2:
        raise ValueError("Empty or degenerate foreground mask; check bumper selection")
    center = ((xx.min() + xx.max()) / 2, (yy.min() + yy.max()) / 2)
    side = max(4, int(max(np.ptp(xx), np.ptp(yy)) * padding))
    box = tuple(int(round(v)) for v in (center[0]-side//2, center[1]-side//2,
                                       center[0]+side//2, center[1]+side//2))
    image.save(output_dir / "foreground_rgba.png")
    cropped = image.crop(box)
    cropped.getchannel("A").save(output_dir / "input_mask.png")
    rgb = Image.new("RGB", cropped.size, (0, 0, 0))
    rgb.paste(cropped.convert("RGB"), mask=cropped.getchannel("A"))
    rgb.save(output_dir / "model_input.png")
    write_json(output_dir / "preprocess.json", {
        "original_size": original_size, "resized_size": resized, "crop_xyxy": box,
        "model_input_size": rgb.size, "background_rgb": [0, 0, 0],
        "original_to_model": [[resized[0]/original_size[0], 0, -box[0]],
                              [0, resized[1]/original_size[1], -box[1]], [0, 0, 1]],
        "provided_alpha": bool(valid_alpha), "padding": padding,
    })
    return rgb
