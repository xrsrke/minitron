import torch

from .utils import (
    get_model_parallel_world_size,
    get_model_parallel_group,
    get_model_parallel_rank
)


def _all_reduce(input):
    if get_model_parallel_world_size() == 1:
        return input

    torch.distributed.all_reduce(input, group=get_model_parallel_group)

    return input


def _gather(input):
    world_size = get_model_parallel_world_size()
    if world_size == 1:
        return input

    last_dim_idx = input.ndim - 1
    rank = get_model_parallel_rank()


class _CopyToModelParallelRegion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input

    @staticmethod
    def backward(ctx, grad_output):
        return _all_reduce(grad_output)


class _GatherFromModelParallelRegion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input


def copy_to_model_parallel_region(input):
    return _CopyToModelParallelRegion.apply(input)


def gather_from_model_parallel_region(input):
    return _GatherFromModelParallelRegion.apply(input)
