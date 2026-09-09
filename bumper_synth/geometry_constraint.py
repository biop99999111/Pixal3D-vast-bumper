"""Experimental front-part support constraint; this is not semantic segmentation."""
from collections import deque
import math
import numpy as np
from PIL import Image, ImageFilter


def outer_silhouette(mask):
    """Fill enclosed grille holes for support selection, leaving image alpha intact."""
    foreground = np.asarray(mask.convert("L")) > 127
    outside = np.zeros_like(foreground)
    height, width = foreground.shape
    queue = deque()
    for y, x in [(y, x) for y in range(height) for x in (0, width - 1)] + [
            (y, x) for x in range(width) for y in (0, height - 1)]:
        if not foreground[y, x] and not outside[y, x]:
            outside[y, x] = True
            queue.append((y, x))
    while queue:
        y, x = queue.popleft()
        for yy, xx in ((y-1, x), (y+1, x), (y, x-1), (y, x+1)):
            if 0 <= yy < height and 0 <= xx < width and not foreground[yy, xx] and not outside[yy, xx]:
                outside[yy, xx] = True
                queue.append((yy, xx))
    return Image.fromarray((~outside).astype(np.uint8) * 255)


class FrontPartConstraint:
    def __init__(self, mask_path, camera, settings):
        import torch
        mask = Image.open(mask_path).convert("L")
        if mask.width != mask.height:
            raise ValueError("Support mask must use square model-input coordinates")
        margin = settings.get("silhouette_margin_pixels", 3)
        silhouette = outer_silhouette(mask)
        if margin:
            silhouette = silhouette.filter(ImageFilter.MaxFilter(2 * margin + 1))
        self.mask = torch.from_numpy(np.asarray(silhouette).copy() > 127)
        self.camera = camera
        self.z_min, self.z_max = settings["depth_range"]

    def __call__(self, coords, resolution, stage):
        import torch
        # Same linspace, rotation and front-camera convention as ProjGrid.
        xyz = coords[:, 1:].float() / (resolution - 1) - .5
        xyz = xyz / self.camera.get("mesh_scale", 1.)
        x, y, z = xyz.unbind(dim=1)
        distance = self.camera["distance"] - z
        size = self.mask.shape[0]
        focal = size / (2 * math.tan(self.camera["camera_angle_x"] / 2))
        u = (size / 2 + focal * x / distance).round().long()
        v = (size / 2 - focal * y / distance).round().long()
        valid = (distance > 0) & (u >= 0) & (u < size) & (v >= 0) & (v < size)
        valid &= (z >= self.z_min) & (z <= self.z_max)
        mask = self.mask.to(coords.device)
        valid &= mask[v.clamp(0, size-1), u.clamp(0, size-1)]
        result = coords[valid]
        print(f"[Bumper support] {stage}: {len(coords)} -> {len(result)} tokens; depth={self.z_min},{self.z_max}")
        if len(result) < 16:
            raise ValueError("Bumper support retained fewer than 16 tokens; check mask/camera/depth range")
        return result
