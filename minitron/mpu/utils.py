from typing import Optional

import torch


_MODEL_PARALLEL_GROUP = None


def get_model_parallel_group() -> Optional[int]:
    return _MODEL_PARALLEL_GROUP


def get_model_parallel_world_size() -> int:
    return torch.distributed.get_world_size(group=get_model_parallel_group())


def get_model_parallel_rank() -> int:
    return torch.distributed.get_rank(group=get_model_parallel_group())
