from pathlib import Path

from PIL import Image, ImageDraw

from .common import ROOT, read_json
from .dataset import blender_run


def preview(asset, output, blender):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    cfg = read_json(ROOT / "configs/vast/render.json")
    cfg.update(resolution=384, samples=16, elevation_degrees=[0,0])
    sheet = Image.new("RGB", (768,808), "white")
    draw = ImageDraw.Draw(sheet)
    for index, angle in enumerate((0,90,180,270)):
        directory = output / str(angle)
        directory.mkdir(exist_ok=True)
        blender_run(blender, {"asset": str(Path(asset).resolve()), "output": str(directory),
                             "seed":42, "class_id":-1, "config":cfg, "azimuth":angle}, directory / "job.json")
        x, y = (index%2)*384, (index//2)*404
        sheet.paste(Image.open(directory / "rgb.png"), (x,y))
        draw.text((x+8,y+386),f"View {angle} degrees",fill="black")
    sheet.save(output / "turntable.png")

