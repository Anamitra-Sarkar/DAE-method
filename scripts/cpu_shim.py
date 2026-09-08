"""CPU shim for CPU-only Codespaces — device only, no method change.
Patches torch .cuda()/set_device/seed to no-ops on CPU so upstream
MedIAnomaly code (written for CUDA) runs unchanged mathematically.
DAE noise values, U-Net arch, loss are untouched.
"""
import torch

_orig_module_cuda = torch.nn.Module.cuda
def _module_cuda_noop(self, *a, **k):
    if torch.cuda.is_available():
        return _orig_module_cuda(self, *a, **k)
    return self
torch.nn.Module.cuda = _module_cuda_noop

_orig_tensor_cuda = torch.Tensor.cuda
def _tensor_cuda_noop(self, *a, **k):
    if torch.cuda.is_available():
        return _orig_tensor_cuda(self, *a, **k)
    return self
torch.Tensor.cuda = _tensor_cuda_noop

if not torch.cuda.is_available():
    torch.cuda.set_device = lambda *a, **k: None
    torch.cuda.manual_seed = lambda *a, **k: None
    torch.cuda.manual_seed_all = lambda *a, **k: None

def resolve_map_location(gpu_id=0):
    if torch.cuda.is_available():
        return torch.device(f'cuda:{gpu_id}')
    return torch.device('cpu')
