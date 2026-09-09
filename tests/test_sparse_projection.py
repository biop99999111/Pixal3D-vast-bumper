"""CPU numerical checks without downloading DINO/NAF or CUDA extensions."""
import ast
import __future__
from types import SimpleNamespace
from pathlib import Path
from typing import Optional, Tuple
import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F


SOURCE = Path(__file__).resolve().parents[1] / "pixal3d/trainers/flow_matching/mixins/image_conditioned_proj.py"
tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
names = {"project_points_to_image_batch", "sample_features", "ProjGrid", "DinoV3ProjFeatureExtractor"}
namespace = dict(torch=torch, nn=nn, F=F, Optional=Optional, Tuple=Tuple)
exec(compile(ast.Module(body=[n for n in tree.body if getattr(n, "name", None) in names],
                        type_ignores=[]), str(SOURCE), "exec",
             flags=__future__.annotations.compiler_flag), namespace)
ProjGrid = namespace["ProjGrid"]


class SparseProjectionTests(unittest.TestCase):
    def test_matches_dense_selection(self):
        torch.manual_seed(42)
        for resolution in (4, 8, 96):
            grid = ProjGrid(resolution, 64)
            # Include corners, duplicates and non-sorted coordinates.
            indices = torch.tensor([resolution**3 - 1, 0, 13, 2, 13, resolution**3 // 2])
            for bhwc in (True, False):
                features = torch.randn(2, 9, 11, 3)
                if not bhwc:
                    features = features.permute(0, 3, 1, 2).contiguous()
                args = (features, torch.tensor([.8, 1.1]),
                        torch.tensor([2., 2.4]), torch.tensor([1., .8]))
                dense = grid(*args, BHWC=bhwc)[:, indices]
                for chunk_size in (1, 4, 8192):
                    sparse = grid(*args, BHWC=bhwc, point_indices=indices,
                                  chunk_size=chunk_size)
                    torch.testing.assert_close(sparse, dense, rtol=1e-5, atol=1e-5)

    def test_chunk_limit(self):
        grid = ProjGrid(4, 64)
        with self.assertRaises(ValueError):
            grid(torch.zeros(1, 3, 3, 2), torch.ones(1), torch.ones(1),
                 torch.ones(1), point_indices=torch.tensor([0]), chunk_size=0)

    def test_extractor_branches(self):
        extractor_class = namespace["DinoV3ProjFeatureExtractor"]
        for use_naf in (False, True):
            extractor = extractor_class.__new__(extractor_class)
            nn.Module.__init__(extractor)
            extractor.use_naf_upsample = use_naf
            extractor.transform = lambda x: x
            extractor.model = SimpleNamespace(config=SimpleNamespace(num_register_tokens=4))
            extractor.patch_number = 2
            extractor.proj_grid = ProjGrid(8, 32)
            features = torch.randn(1, 9, 3)
            extractor.extract_features = lambda x: features
            extractor._load_naf = lambda: None
            extractor.naf_target_size = 8
            extractor.naf_model = lambda guide, lr, size: F.interpolate(lr, size=size, mode="bilinear")
            args = (torch.randn(1, 3, 8, 8), torch.tensor([.8]),
                    torch.tensor([2.]), torch.tensor([1.]))
            indices = torch.tensor([511, 0, 100, 25, 100])
            global_dense, dense = extractor(*args)
            global_sparse, sparse = extractor(*args, point_indices=indices, projection_chunk_size=2)
            torch.testing.assert_close(global_sparse, global_dense)
            torch.testing.assert_close(sparse, dense[:, indices], atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
