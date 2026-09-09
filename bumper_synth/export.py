from pathlib import Path

import numpy as np

from .common import write_json

# Same coordinate transform used by the supplied inference.py.
EXPORT_TRANSFORM = np.array([[-1, 0, 0, 0], [0, 0, -1, 0],
                             [0, -1, 0, 0], [0, 0, 0, 1]], dtype=np.float64)


def save_mesh_state(mesh, resolution, layout, path):
    def cpu(value):
        import torch
        value = value.detach().cpu()
        if value.dtype == torch.bfloat16:
            value = value.float()
        return value.numpy()
    np.savez_compressed(path, vertices=cpu(mesh.vertices), faces=cpu(mesh.faces),
                        attrs=cpu(mesh.attrs), coords=cpu(mesh.coords), resolution=int(resolution))
    write_json(Path(path).with_suffix(".layout.json"),
               {key: [value.start, value.stop] for key, value in layout.items()})


def export_state(state_path, output_path, settings):
    import torch
    import o_voxel
    from .common import read_json
    if settings["texture_size"] not in (1024, 2048, 4096, 8192):
        raise ValueError("Unsupported texture size")
    if settings["decimation_target"] < 1000:
        raise ValueError("decimation_target must be >= 1000")
    data = np.load(state_path, allow_pickle=False)
    layout = {key: slice(*value) for key, value in
              read_json(Path(state_path).with_suffix(".layout.json")).items()}
    mesh = o_voxel.postprocess.to_glb(
        vertices=torch.from_numpy(data["vertices"]).cuda(),
        faces=torch.from_numpy(data["faces"]).cuda(),
        attr_volume=torch.from_numpy(data["attrs"]).cuda(),
        coords=torch.from_numpy(data["coords"]).cuda(), attr_layout=layout,
        grid_size=int(data["resolution"]), aabb=[[-.5, -.5, -.5], [.5, .5, .5]],
        decimation_target=settings["decimation_target"], texture_size=settings["texture_size"],
        remesh=settings["remesh"], remesh_band=1, remesh_project=0, use_tqdm=True)
    mesh.apply_transform(EXPORT_TRANSFORM)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(output_path), extension_webp=settings.get("webp", False))
    write_json(Path(output_path).with_suffix(".export.json"),
               {"settings": settings, "transform": EXPORT_TRANSFORM.tolist()})
