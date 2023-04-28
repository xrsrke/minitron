from datetime import timedelta

import torch

from .initialization import initialize_model_parallel


class BertModel:
    pass


def model_provider(pre_process=True, post_process=True):
    model = BertModel(
        pre_process=pre_process,
        post_process=post_process
    )

    return model


def pretrain(dataset, model):
    # pretrain_bert.py
    pass


_TENSOR_MODEL_PARALLEL_GROUP = None
_PIPELINE_MODEL_PARALLEL_GROUP = None
_DATA_PARALLEL_GROUP = None


def is_model_parallel_initialized():
    # https://github.com/NVIDIA/Megatron-LM/blob/3db2063b1ff992a971ba18f7101eecc9c4e90f03/megatron/core/parallel_state.py#L241
    if _TENSOR_MODEL_PARALLEL_GROUP is None or \
            _PIPELINE_MODEL_PARALLEL_GROUP is None or \
            _DATA_PARALLEL_GROUP is None:
        return False
    return True


def initialize_distributed():
    # https://github.com/NVIDIA/Megatron-LM/blob/3db2063b1ff992a971ba18f7101eecc9c4e90f03/megatron/initialize.py#L147
    args = {}
    device_count = torch.cuda.device_count()

    if torch.distributed.is_initialized():
        args.rank = torch.distributed.get_rank()
        args.world_size = torch.distributed.get_world_size()
    else:
        # TODO: why need this?
        if device_count > 0:
            device = args.rank % device_count
            if args.local_rank is not None:
                assert args.local_rank == device, \
                    "expected local-rank to be the same rank % device_count"
            else:
                args.local_rank = device
            torch.cuda.set_device(device)

    # call the init process
    torch.distributed.init_process_group(
        backend=args.distributed_backend,
        world_size=args.world_size, rank=args.rank,
        timeout=timedelta(seconds=300)
    )

    if device_count > 0:
        if is_model_parallel_initialized() is True:
            print("model parallel already initialized")
        else:
            initialize_model_parallel(
                args.tensor_model_parallel_size,
                args.pipeline_model_parallel_size,
                args.virtual_pipeline_model_parallel_size,
                args.pipeline_model_parallel_split_rank
            )
