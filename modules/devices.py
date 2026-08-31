import contextlib
import torch
from backend import memory_management


def has_xpu() -> bool:
    return memory_management.xpu_available


def has_mps() -> bool:
    return memory_management.mps_mode()


def cuda_no_autocast(device_id=None) -> bool:
    return False


def get_cuda_device_id():
    return memory_management.get_torch_device().index


def get_cuda_device_string():
    return str(memory_management.get_torch_device())


def get_optimal_device_name():
    return memory_management.get_torch_device().type


def get_optimal_device():
    return memory_management.get_torch_device()


def get_device_for(task):
    return memory_management.get_torch_device()


def torch_gc():
    memory_management.soft_empty_cache()


def torch_npu_set_device():
    return


def enable_tf32():
    return


cpu: torch.device = torch.device("cpu")
fp8: bool = False
device: torch.device = memory_management.get_torch_device()
device_interrogate: torch.device = memory_management.text_encoder_device()  # for backward compatibility, not used now
device_gfpgan: torch.device = memory_management.get_torch_device()  # will be managed by memory management system
device_esrgan: torch.device = memory_management.get_torch_device()  # will be managed by memory management system
device_codeformer: torch.device = memory_management.get_torch_device()  # will be managed by memory management system
dtype: torch.dtype = torch.float32 if memory_management.unet_dtype() is torch.float32 else torch.float16
dtype_vae: torch.dtype = memory_management.vae_dtype()
dtype_unet: torch.dtype = memory_management.unet_dtype()
dtype_inference: torch.dtype = memory_management.unet_dtype()
unet_needs_upcast = False


def cond_cast_unet(input):
    return input


def cond_cast_float(input):
    return input


nv_rng = None
patch_module_list = []


def manual_cast_forward(target_dtype):
    return


@contextlib.contextmanager
def manual_cast(target_dtype):
    return


def autocast(disable=False):
    return contextlib.nullcontext()


def without_autocast(disable=False):
    return contextlib.nullcontext()


class NansException(Exception):
    pass


def test_for_nans(x, where):
    # This was stubbed out during the port, which turned every NaN'd run into
    # a silent black image. Restored 2026-08-29 after chasing exactly that: a
    # completed 14/14 run under heavy partial-offload delivering pure black
    # with a clean console. Cheap check (first element only -- the all-NaN
    # collapse this exists to catch makes every element NaN).
    from modules import shared
    if getattr(shared.opts, 'disable_nan_check', False):
        return
    if not torch.isnan(x[(0,) * len(x.shape)]):
        return
    if where == "unet":
        message = ("A tensor with all NaNs was produced in the UNet -- the image would have "
                   "come out solid black. On this setup the usual cause is arithmetic "
                   "degrading under heavy memory pressure (deep CPU-swap of the model "
                   "weights); free VRAM -- close other GPU apps, stop the assistant LLM, or "
                   "lower resolution -- and retry.")
    elif where == "vae":
        message = ("A tensor with all NaNs was produced in the VAE -- the image would have "
                   "come out solid black. Free VRAM and retry.")
    else:
        message = f"A tensor with all NaNs was produced in {where}."
    raise NansException(message)


def first_time_calculation():
    return
