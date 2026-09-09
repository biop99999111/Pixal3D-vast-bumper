import numpy as np
import pytest
import torch
from PIL import Image

from bumper_synth.geometry_constraint import FrontPartConstraint, outer_silhouette
from test_sparse_projection import namespace, ProjGrid


def test_outer_silhouette_fills_grille_not_exterior():
    mask = np.zeros((20, 30), dtype=np.uint8)
    mask[5:15, 4:26] = 255
    mask[8:12, 10:20] = 0
    result = np.asarray(outer_silhouette(Image.fromarray(mask)))
    assert result[10, 15] == 255
    assert result[0, 15] == 0
    assert mask[10, 15] == 0


def test_support_matches_upstream_camera_projection(tmp_path):
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[24:43, 5:59] = 255
    path = tmp_path / "mask.png"
    Image.fromarray(mask).save(path)
    camera = {"camera_angle_x": .8, "distance": 1.5, "mesh_scale": 1.}
    settings = {"depth_range": [.15, .5], "silhouette_margin_pixels": 0}
    constraint = FrontPartConstraint(path, camera, settings)
    resolution = 32
    xyz = torch.cartesian_prod(*[torch.arange(resolution)] * 3)
    coords = torch.cat([torch.zeros(len(xyz), 1, dtype=torch.long), xyz], dim=1)
    actual = constraint(coords, resolution, "test")
    grid = ProjGrid(resolution, 64)
    transform = grid.front_view_transform_matrix[None].clone()
    transform[:, 1, 3] = -camera["distance"]
    points, _, _ = namespace["project_points_to_image_batch"](
        grid.grid_points[None] / 2, transform, torch.tensor([.8]), 64)
    u, v = points[0].round().long().unbind(dim=1)
    z = xyz[:, 2].float() / 31 - .5
    expected = (u >= 5) & (u < 59) & (v >= 24) & (v < 43) & (z >= .15) & (z <= .5)
    assert torch.equal(actual, coords[expected])
    assert len(actual) < len(coords) / 3


def test_empty_support_does_not_fall_back_to_whole_car(tmp_path):
    path = tmp_path / "empty.png"
    Image.new("L", (32, 32)).save(path)
    constraint = FrontPartConstraint(path, {"camera_angle_x": .8, "distance": 1.5},
                                     {"depth_range": [.15, .5]})
    with pytest.raises(ValueError, match="fewer than 16"):
        constraint(torch.zeros(30, 4, dtype=torch.int32), 32, "test")
