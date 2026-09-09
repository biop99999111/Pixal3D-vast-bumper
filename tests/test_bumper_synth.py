import numpy as np
import pytest
from PIL import Image

from bumper_synth.common import digest, read_json, write_json
from bumper_synth.preprocess import prepare
from bumper_synth.labels import mask_to_box, write_labels
from bumper_synth.quality import approve_asset, require_review
from bumper_synth.dataset import assemble
from bumper_synth.alignment import camera_to_blender


def test_mask_preprocess_preserves_hole_and_transform(tmp_path):
    rgba = np.zeros((80,160,4), dtype=np.uint8)
    rgba[20:60,20:140] = [200,40,20,255]
    rgba[30:50,70:90] = 0
    path = tmp_path / "input.png"
    Image.fromarray(rgba).save(path)
    def forbidden(_):
        raise AssertionError("Background remover must not execute")
    image = prepare(path, tmp_path / "out", "provided_mask", remover=forbidden)
    metadata = read_json(tmp_path / "out/preprocess.json")
    x, y, _ = np.array(metadata["original_to_model"]) @ [80,40,1]
    assert image.getpixel((int(x),int(y))) == (0,0,0)
    assert image.width == image.height


def test_empty_and_opaque_mask_rejected(tmp_path):
    path = tmp_path / "image.png"
    Image.new("RGBA", (32,32), (255,0,0,0)).save(path)
    with pytest.raises(ValueError, match="Empty"):
        prepare(path, tmp_path / "empty", "provided_mask")
    Image.new("RGB", (32,32)).save(path)
    with pytest.raises(ValueError, match="provided_mask"):
        prepare(path, tmp_path / "opaque", "provided_mask")


def test_boundary_box_covers_exact_visible_pixels(tmp_path):
    mask = np.zeros((50,100), dtype=bool)
    mask[40:50,90:100] = True
    box = mask_to_box(mask)
    assert box == pytest.approx([.95,.9,.1,.2])
    write_labels(tmp_path / "label.txt", [[2,*box]])
    assert (tmp_path / "label.txt").read_text().startswith("2 ")
    assert mask_to_box(np.zeros((10,10),dtype=bool)) is None
    assert mask_to_box(np.eye(3,dtype=bool)) is None


def test_review_invalidated_by_changed_asset(tmp_path):
    asset = tmp_path / "asset.glb"
    asset.write_bytes(b"original")
    with pytest.raises(ValueError, match="Review"):
        require_review(asset)
    approve_asset(asset, "Checked clean bumper")
    require_review(asset)
    asset.write_bytes(b"modified")
    with pytest.raises(ValueError, match="changed"):
        require_review(asset)


def test_assembly_rejects_group_leakage_and_fixtures(tmp_path):
    paths = []
    for split in ("train","val"):
        p = tmp_path / split
        write_json(p / "dataset.json", {"status":"complete", "demo_only":False,
                   "group":"same-car", "asset_sha256":split, "split":split, "samples":[]})
        paths.append(p)
    with pytest.raises(ValueError, match="leaked"):
        assemble(paths, tmp_path / "assembled")
    cfg = read_json(paths[0] / "dataset.json")
    cfg["demo_only"] = True
    write_json(paths[0] / "dataset.json", cfg)
    with pytest.raises(ValueError, match="fixture"):
        assemble(paths, tmp_path / "assembled")


def test_camera_transform_preserves_voxel_projection():
    grid = np.array([[1,0,0,0],[0,0,-1,0],[0,1,0,0],[0,0,0,1]],dtype=float)
    export = np.array([[-1,0,0,0],[0,0,-1,0],[0,-1,0,0],[0,0,0,1]],dtype=float)
    original = grid.copy(); original[1,3] = -3
    camera = camera_to_blender({"distance":3,"export_transform":export.tolist()})
    point = np.array([.15,.2,.1,1])
    before = np.linalg.inv(original) @ grid @ point
    after = np.linalg.inv(camera) @ grid @ export @ point
    assert after == pytest.approx(before)


def test_yolo_rejects_nonfinite_or_outside_box(tmp_path):
    for row in ([0,float('nan'),.5,.1,.1],[0,.95,.5,.3,.1],[4,.5,.5,.1,.1]):
        with pytest.raises(ValueError):
            write_labels(tmp_path / "bad.txt", [row])
