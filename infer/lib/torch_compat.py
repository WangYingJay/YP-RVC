"""让本机训练权重按旧方式加载，同时不再每次 torch.load 都打印警告。"""


def allow_legacy_torch_load():
    import torch

    if getattr(torch.load, "_rvc_legacy_load", False):
        return
    original = torch.load

    def load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original(*args, **kwargs)

    load._rvc_legacy_load = True
    torch.load = load
