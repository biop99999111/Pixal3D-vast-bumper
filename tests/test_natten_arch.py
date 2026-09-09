import pytest
import os
import subprocess
import sys
from pathlib import Path

from scripts.natten_arch import format_arch


@pytest.mark.parametrize("capability, expected", [((12,0),"12.0"),((8,9),"8.9"),((10,0),"10.0")])
def test_natten_receives_compute_capability_not_nvcc_tag(capability, expected):
    value = format_arch(capability)
    assert value == expected
    # NATTEN v0.21.0's setup parser multiplies the supplied float by ten.
    assert int(float(value)*10) == capability[0]*10+capability[1]


@pytest.mark.parametrize("bad", ["120","1200","sm_120","12.0;89","12.0+PTX"])
def test_invalid_inherited_override_stops_before_compilation(bad):
    with pytest.raises(ValueError, match="dotted"):
        format_arch((12,0), bad)


def test_explicit_multi_gpu_override():
    assert format_arch((12,0), "8.9;12.0") == "8.9;12.0"


def test_image_build_override_needs_no_torch_or_gpu(tmp_path):
    # Fail immediately if the CLI tries to import Torch during an explicit build.
    (tmp_path / "torch.py").write_text("raise RuntimeError('No GPU discovery in image builds')")
    script = Path(__file__).resolve().parents[1] / "scripts" / "natten_arch.py"
    env = dict(os.environ, NATTEN_CUDA_ARCH="8.9;12.0", PYTHONPATH=str(tmp_path))
    result = subprocess.run([sys.executable, str(script)], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "8.9;12.0"
