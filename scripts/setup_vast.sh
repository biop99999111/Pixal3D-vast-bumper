#!/usr/bin/env bash
set -euo pipefail
# Run from a Linux CUDA development image. Do not downgrade a working 5090 Torch stack.
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
export MAX_JOBS="${MAX_JOBS:-4}"
export NATTEN_N_WORKERS="${NATTEN_N_WORKERS:-4}"
if [[ "${1:-}" == "--help" ]]; then
  echo 'Usage: bash scripts/setup_vast.sh [--lightweight|--resume-natten]'
  echo '--resume-natten skips completed dependency builds and resumes at NATTEN.'
  echo 'Full setup requires Linux, git, a C++ compiler, nvcc and a CUDA-capable PyTorch/torchvision installation.'
  echo 'Use a CUDA 12.8+ development image for RTX 5090. Existing Torch is preserved.'
  exit 0
fi
case "${1:-}" in
  ''|--lightweight|--resume-natten) ;;
  *) echo "Unknown option: $1"; exit 2 ;;
esac
if [[ "${1:-}" != "--resume-natten" ]]; then
python -m pip install -r requirements-vast.txt
if [[ "${1:-}" == "--lightweight" ]]; then exit 0; fi
command -v nvcc >/dev/null || { echo 'nvcc missing: select a CUDA development image, not runtime-only.'; exit 1; }
python - <<'PY'
import torch, torchvision
assert torch.cuda.is_available(), 'CUDA is unavailable'
major, minor = torch.cuda.get_device_capability()
runtime = tuple(map(int, torch.version.cuda.split('.')[:2]))
if major >= 12 and runtime < (12, 8):
    raise SystemExit('Blackwell requires a suitable CUDA 12.8+ Torch build; use another template.')
x = torch.ones((32,32), device='cuda'); (x @ x).sum().item()
print('Keeping installed Torch:', torch.__version__, 'GPU:', torch.cuda.get_device_name())
PY
python -m pip freeze > cache-before-install.txt
python -m pip install setuptools wheel ninja cmake
mkdir -p cache/build
# Record exact checked-out refs and reuse them on subsequent installs.
checkout_dep() {
  local url="$1" target="$2" ref="${3:-}"
  if [[ ! -d "$target/.git" ]]; then
    git clone --recursive "$url" "$target"
    if [[ -n "$ref" ]]; then git -C "$target" checkout "$ref"; fi
    git -C "$target" submodule update --init --recursive
  fi
  if [[ -n "$ref" ]] && [[ "$(git -C "$target" rev-parse HEAD)" != "$(git -C "$target" rev-parse "$ref^{commit}")" ]]; then
    echo "Dependency checkout differs from lock: $target. Use a fresh cache/build directory."
    exit 1
  fi
  git -C "$target" rev-parse HEAD
}
locked_ref() {
  python -c 'import json,sys; print(json.load(open("configs/vast/source-refs.json"))[sys.argv[1]])' "$1"
}
checkout_dep https://github.com/microsoft/TRELLIS.2.git cache/build/trellis2 "$(locked_ref trellis2)"
checkout_dep https://github.com/JeffreyXiang/CuMesh.git cache/build/cumesh "$(locked_ref cumesh)"
checkout_dep https://github.com/JeffreyXiang/FlexGEMM.git cache/build/flexgemm "$(locked_ref flexgemm)"
checkout_dep https://github.com/NVlabs/nvdiffrast.git cache/build/nvdiffrast "$(locked_ref nvdiffrast)"
checkout_dep https://github.com/microsoft/MoGe.git cache/build/moge "$(locked_ref moge)"
python - <<'PY'
from pathlib import Path
lines=Path('requirements.txt').read_text().splitlines()
Path('cache/build/requirements-pixal.txt').write_text('\n'.join(x for x in lines if not x.startswith('git+'))+'\n')
import torch, torchvision
Path('cache/build/torch-constraints.txt').write_text(f'torch=={torch.__version__}\ntorchvision=={torchvision.__version__}\n')
PY
export PIP_CONSTRAINT="$repo_dir/cache/build/torch-constraints.txt"
python -m pip install -r cache/build/requirements-pixal.txt
python -m pip install cache/build/moge
python -m pip install cache/build/nvdiffrast --no-build-isolation
python -m pip install cache/build/cumesh --no-build-isolation
python -m pip install cache/build/flexgemm --no-build-isolation
python -m pip install cache/build/trellis2/o-voxel --no-build-isolation
fi
# Assignment must be separate from export so Python validation failures stop the shell.
NATTEN_CUDA_ARCH="$(python scripts/natten_arch.py)"
export NATTEN_CUDA_ARCH
echo "Building NATTEN with compute capability $NATTEN_CUDA_ARCH"
mkdir -p cache/build outputs/setup
if [[ -f cache/build/torch-constraints.txt ]]; then
  export PIP_CONSTRAINT="$repo_dir/cache/build/torch-constraints.txt"
fi
python -m pip install natten==0.21.0 --no-build-isolation --no-deps --verbose 2>&1 | tee outputs/setup/natten-build.log
python -m pip install 'https://github.com/LDYang694/Storages/releases/download/20260430/utils3d-0.0.2-py3-none-any.whl'
python -m pip check
python -m ipykernel install --user --name pixal3d-vast --display-name 'Pixal3D Vast'
python -m pip freeze > cache/requirements-resolved.txt
python - <<'PY'
import json, subprocess
from pathlib import Path
refs={p.name:subprocess.check_output(['git','-C',str(p),'rev-parse','HEAD'],text=True).strip()
      for p in Path('cache/build').iterdir() if (p/'.git').exists()}
Path('cache/build-refs.json').write_text(json.dumps(refs,indent=2))
PY
ATTN_BACKEND=sdpa python scripts/check_environment.py --gpu-smoke
echo 'Setup complete. Install Blender separately, then open notebooks/workspace.ipynb.'
