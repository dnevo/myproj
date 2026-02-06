import os, torch

def get_device(prefer_gpu=True):
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    if prefer_gpu and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def device_report(device):
    parts = [f"device={device}"]
    if device.type == "cuda":
        parts.append(torch.cuda.get_device_name(0))
    parts.append(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','')!r}")
    return " | ".join(parts)
