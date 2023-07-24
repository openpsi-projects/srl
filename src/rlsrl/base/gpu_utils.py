from typing import List
import itertools
import logging
import os
import platform

logger = logging.getLogger("System-GPU")


def gpu_count():
    """Returns the number of gpus on a node. Ad-hoc to frl cluster.
    """
    if platform.system() == "Darwin":
        return 0
    elif platform.system() == "Windows":
        try:
            import torch
            return torch.cuda.device_count()
        except ImportError:
            return 0
    else:
        dev_directories = list(os.listdir("/dev/"))
        for cnt in itertools.count():
            if "nvidia" + str(cnt) in dev_directories:
                continue
            else:
                break
        return cnt


def resolve_cuda_environment():
    """Pytorch DDP does not work if more than one processes (with different environment variable CUDA_VISIBLE_DEVICES)
     are inited on the same node(w/ multiple GPUS). This function works around the issue by setting another variable.
     Currently all devices should use `base.gpu_utils.get_gpu_device()` to get the proper gpu device.
    """
    if "MARL_CUDA_DEVICES" in os.environ.keys():
        return

    cuda_devices = [str(i) for i in range(gpu_count())]
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        if len(cuda_devices) > 0:
            os.environ["MARL_CUDA_DEVICES"] = "0"
        else:
            os.environ["MARL_CUDA_DEVICES"] = "cpu"
    else:
        if os.environ["CUDA_VISIBLE_DEVICES"] != "":
            for s in os.environ["CUDA_VISIBLE_DEVICES"].split(","):
                assert s.isdigit() and s in cuda_devices, f"Cuda device {s} cannot be resolved."
            os.environ["MARL_CUDA_DEVICES"] = os.environ["CUDA_VISIBLE_DEVICES"]  # Store assigned device.
        else:
            os.environ["MARL_CUDA_DEVICES"] = "cpu"  # Use CPU if no cuda device available.
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(cuda_devices)  # Make all devices visible.


def get_gpu_device() -> List[str]:
    """
    Returns:
        List of assigned devices.
    """
    if "MARL_CUDA_DEVICES" not in os.environ:
        resolve_cuda_environment()

    if os.environ["MARL_CUDA_DEVICES"] == "cpu":
        return ["cpu"]
    else:
        return [f"cuda:{device}" for device in os.environ["MARL_CUDA_DEVICES"].split(",")]


def set_cuda_device(device):
    """Set the default cuda-device. Useful on multi-gpu nodes. Should be called in every gpu-thread.
    """
    logger.info(f"Setting device to {device}.")
    if device != "cpu":
        import torch
        torch.cuda.set_device(device)


resolve_cuda_environment()
