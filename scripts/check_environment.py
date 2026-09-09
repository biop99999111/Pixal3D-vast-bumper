"""Read-only diagnostics. --gpu-smoke executes real CUDA kernels, not just imports."""
import argparse
import importlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bumper_synth.common import safe_error, write_json


def module_version(obj, module):
    # Lazy packages may interpret __version__ as a submodule import. Reading
    # the namespace directly avoids invoking their module-level __getattr__.
    version = vars(obj).get("__version__")
    if version is not None:
        return str(version)
    distribution = {"PIL": "Pillow", "nvdiffrast.torch": "nvdiffrast"}.get(module, module)
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-smoke", action="store_true")
    parser.add_argument("--output", default="outputs/environment.json")
    args = parser.parse_args()
    result = {"python": sys.version, "platform": platform.platform(),
              "free_disk_bytes": shutil.disk_usage(Path.cwd()).free, "modules": {}, "checks": {}}
    failed = False
    for module in ("torch", "torchvision", "natten", "cumesh", "o_voxel", "flex_gemm",
                   "nvdiffrast.torch", "transformers", "moge", "utils3d", "einops", "PIL"):
        try:
            obj = importlib.import_module(module)
            result["modules"][module] = {"ok": True, "version": module_version(obj, module)}
        except Exception as error:
            result["modules"][module] = {"ok": False, "error": safe_error(error)}
            failed = True
    for command in ("nvidia-smi", "nvcc", "blender"):
        path = shutil.which(command)
        result[command] = path
    try:
        import torch
        result["cuda"] = {"available": torch.cuda.is_available(), "runtime": torch.version.cuda,
                          "arch_list": torch.cuda.get_arch_list()}
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            result["cuda"].update(name=props.name, capability=list(torch.cuda.get_device_capability()),
                                  total_memory=props.total_memory)
        if args.gpu_smoke:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA unavailable")
            a = torch.randn(64, 64, device="cuda")
            assert torch.isfinite(a @ a).all()
            torch.nn.functional.scaled_dot_product_attention(
                a.reshape(1, 1, 64, 64), a.reshape(1, 1, 64, 64), a.reshape(1, 1, 64, 64))
            torch.cuda.synchronize()
            result["checks"]["torch_sdpa"] = "passed"
            import natten
            q = torch.randn(1, 8, 8, 2, 32, device="cuda", dtype=torch.float16)
            natten.functional.na2d(q, q, q, kernel_size=3)
            torch.cuda.synchronize()
            result["checks"]["natten"] = "passed"
            import nvdiffrast.torch as dr
            context = dr.RasterizeCudaContext()
            positions = torch.tensor([[[-.5,-.5,0,1],[.5,-.5,0,1],[0,.5,0,1]]], device="cuda")
            triangles = torch.tensor([[0,1,2]], dtype=torch.int32, device="cuda")
            dr.rasterize(context, positions, triangles, resolution=[32,32])
            torch.cuda.synchronize()
            result["checks"]["nvdiffrast"] = "passed"
    except Exception as error:
        failed = True
        result["checks"]["error"] = safe_error(error)
    result["passed"] = not failed
    result["scope"] = "Imports plus requested CUDA probes; full Pixal3D/export still requires inference smoke test."
    write_json(args.output, result)
    print(json.dumps(result, indent=2))
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
