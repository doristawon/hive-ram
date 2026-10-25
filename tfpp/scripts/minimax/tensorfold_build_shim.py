"""Minimal stand-in for TensorFold 0.6.6's build.load: torch JIT for this GPU only."""
import os
def load(name, sources, need=None, *, arch_specific=False, **kw):
    import torch
    from torch.utils import cpp_extension
    ma, mi = torch.cuda.get_device_capability()
    flags = list(kw.pop("extra_cuda_cflags", [])) + [f"-gencode=arch=compute_{ma}{mi},code=sm_{ma}{mi}"]
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", f"{ma}.{mi}")
    return cpp_extension.load(name=name, sources=sources, extra_cuda_cflags=flags,
                              build_directory=os.environ.get("TFX_BUILD") or None, **kw)
