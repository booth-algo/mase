# MASE adaptation: JIT-compile the torch_dwn EFD CUDA extension at runtime.
# This avoids ABI mismatch issues from pre-compiled wheels.
#
# Source kernel and bridge files copied from torch_dwn v1.1.1
# (https://github.com/alanbacellar/DWN) by Alan T. L. Bacellar.

import os
import warnings
import torch

_ext = None
_ext_tried = False


def _try_load_cuda_ext():
    global _ext, _ext_tried
    if _ext_tried:
        return _ext
    _ext_tried = True

    if not torch.cuda.is_available():
        return None

    try:
        from torch.utils.cpp_extension import load
        src_dir = os.path.join(os.path.dirname(__file__), "cpp")
        _ext = load(
            name="dwn_efd_cuda",
            sources=[
                os.path.join(src_dir, "efd_cuda.cpp"),
                os.path.join(src_dir, "efd_cuda_kernel.cu"),
            ],
            verbose=False,
        )
        return _ext
    except Exception as e:
        warnings.warn(
            f"DWN: CUDA EFD extension failed to compile ({e}). "
            "GPU training will not work.",
            RuntimeWarning,
        )
        return None


def get_cuda_ext():
    """Return the compiled CUDA extension, or None if unavailable."""
    return _try_load_cuda_ext()
