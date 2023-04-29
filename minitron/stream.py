from typing import Union
from contextlib import contextmanager

import torch


class CPUStreamType:
    pass


CPUStream = CPUStreamType()

AbstractStream = Union[torch.cuda.Stream, CPUStreamType]


def is_cuda(stream: AbstractStream) -> bool:
    return stream is not CPUStream


@contextmanager
def use_stream(stream: AbstractStream):
    if not is_cuda(stream):
        yield
        return

    with torch.cuda.stream(stream):
        yield


@contextmanager
def use_device(device: torch.device):
    if device.type != "cuda":
        yield
        return

    with torch.cuda.device(device):
        yield
