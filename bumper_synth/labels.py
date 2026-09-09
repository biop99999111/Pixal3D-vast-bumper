import numpy as np
from PIL import Image, ImageDraw

CLASSES = {0: "dent", 1: "ding", 2: "scratch"}


def mask_to_box(mask, min_pixels=12, min_side=2):
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("Expected a 2-D visible mask")
    yy, xx = np.where(mask)
    if len(xx) < min_pixels:
        return None
    x0, y0, x1, y1 = int(xx.min()), int(yy.min()), int(xx.max())+1, int(yy.max())+1
    if min(x1-x0, y1-y0) < min_side:
        return None
    height, width = mask.shape
    return [(x0+x1)/(2*width), (y0+y1)/(2*height), (x1-x0)/width, (y1-y0)/height]


def overlay(image_path, rows, output_path):
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    for row in rows:
        cls, x, y, w, h = row
        box = [(x-w/2)*image.width, (y-h/2)*image.height,
               (x+w/2)*image.width, (y+h/2)*image.height]
        draw.rectangle(box, outline="red", width=2)
        draw.text((box[0], max(0, box[1]-12)), CLASSES[int(cls)], fill="red")
    image.save(output_path)


def write_labels(path, rows):
    for cls, x, y, w, h in rows:
        if cls not in CLASSES or not all(np.isfinite([x, y, w, h])):
            raise ValueError("Invalid class or non-finite label")
        if not (0 < w <= 1 and 0 < h <= 1 and w/2-1e-7 <= x <= 1-w/2+1e-7
                and h/2-1e-7 <= y <= 1-h/2+1e-7):
            raise ValueError("Bounding box outside image")
    path.write_text("".join(f"{int(c)} {x:.8f} {y:.8f} {w:.8f} {h:.8f}\n"
                            for c, x, y, w, h in rows), encoding="utf-8")
