import torch
from backend.args import args


def stream_context():
    # NOTE (ROCm): no separate branch is needed here. A ROCm torch build reports
    # torch.cuda.is_available() == True and exposes HIP streams through the
    # torch.cuda API, so the first branch already returns the right thing. (An
    # added `torch.version.hip` branch below these would also be dead code --
    # unreachable on ROCm because the cuda branch returns first.)
    if torch.cuda.is_available():
        return torch.cuda.stream

    if torch.xpu.is_available():
        return torch.xpu.stream

    return None


def get_current_stream():
    try:
        if torch.cuda.is_available():
            device = torch.device(torch.cuda.current_device())
            stream = torch.cuda.current_stream(device)
            with torch.cuda.stream(stream):
                torch.zeros((1, 1)).to(device, torch.float32)
            stream.synchronize()
            return stream
        if torch.xpu.is_available():
            device = torch.device("xpu")
            stream = torch.xpu.current_stream(device)
            with torch.xpu.stream(stream):
                torch.zeros((1, 1)).to(device, torch.float32)
            stream.synchronize()
            return stream
    except:
        return None


def get_new_stream():
    try:
        if torch.cuda.is_available():
            device = torch.device(torch.cuda.current_device())
            stream = torch.cuda.Stream(device)
            with torch.cuda.stream(stream):
                torch.zeros((1, 1)).to(device, torch.float32)
            stream.synchronize()
            return stream
        if torch.xpu.is_available():
            device = torch.device("xpu")
            stream = torch.xpu.Stream(device)
            with torch.xpu.stream(stream):
                torch.zeros((1, 1)).to(device, torch.float32)
            stream.synchronize()
            return stream
    except:
        return None


def should_use_stream():
    return stream_activated and current_stream is not None and mover_stream is not None


current_stream = get_current_stream()
mover_stream = get_new_stream()
stream_activated = args.cuda_stream
