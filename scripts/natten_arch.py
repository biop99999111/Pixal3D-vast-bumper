"""NATTEN expects dotted compute capability, e.g. 12.0, NOT nvcc's 120."""
import os
import re


def format_arch(capability, override=None):
    value = override or ".".join(map(str, capability))
    if not all(re.fullmatch(r"[1-9][0-9]?\.[0-9]", item) for item in value.split(";")):
        raise ValueError(
            "NATTEN_CUDA_ARCH must use dotted values such as '12.0' or '8.9;12.0'. "
            "Values like '120' become invalid compute_1200. Unset the variable or correct it."
        )
    return value


if __name__ == "__main__":
    override = os.getenv("NATTEN_CUDA_ARCH")
    if override:
        print(format_arch(None, override))
    else:
        import torch
        print(format_arch(torch.cuda.get_device_capability()))
