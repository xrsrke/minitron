import os
from copy import deepcopy

import torch
from torch import nn
from torch.multiprocessing import Process
import torch.distributed as dist
import torch.nn.functional as F


from minitron.linear import ColumnParallelLinear, RowParallelLinear

MASTER_ADDR = 'localhost'
MASTER_PORT = '12359'

def run_parallel_column_parallel_linear(
    rank, world_size,
    input_size, output_size,
    input, weight, bias, non_parallel_output,
    weight_grad, bias_grad
):
    os.environ['MASTER_ADDR'] = MASTER_ADDR
    os.environ['MASTER_PORT'] = MASTER_PORT
    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=world_size
    )

    model = ColumnParallelLinear(input_size, output_size, world_size)

    # Partition the weights and biases and assign to the model
    weight_partition_size = weight.shape[0] // world_size
    bias_partition_size = bias.shape[0] // world_size

    model.weight.data = weight[rank*weight_partition_size:(rank+1)*weight_partition_size].detach().requires_grad_(True)
    model.bias.data = bias[rank*bias_partition_size:(rank+1)*bias_partition_size].detach().requires_grad_(True)

    parallel_output = model(input.detach().requires_grad_(False))
    parallel_output.sum(dim=-1).backward()

    assert torch.allclose(parallel_output, non_parallel_output, rtol=1e-3)
    assert torch.allclose(model.weight.grad, weight_grad[rank], rtol=1e-3)
    assert torch.allclose(model.bias.grad, bias_grad[rank], rtol=1e-3)

    dist.destroy_process_group()


def test_column_parallel_linear():
    world_size = 4
    input_size = 16
    output_size = 12

    torch.random.manual_seed(69)

    input = torch.randn(input_size, requires_grad=False)
    weight = torch.randn(output_size, input_size, requires_grad=True)
    bias = torch.randn(output_size, requires_grad=True)

    non_parallel_output = F.linear(input, weight, bias)
    non_parallel_output.sum(dim=-1).backward()

    # because we detach the weight and bias from the computational graph
    # so have to make a copy of the gradients
    weight_grad = weight.grad.clone()
    bias_grad = bias.grad.clone()

    processes = []
    for rank in range(world_size):
        p = Process(target=run_parallel_column_parallel_linear, args=(
            rank, world_size,
            input_size, output_size,
            # because pytorch does not support sending tensors that
            # require gradient through inter-process communication
            # so we gotta detach them from the computational graph
            input, weight.detach(), bias.detach(), non_parallel_output.detach(),
            weight_grad, bias_grad
        ))
        p.start()

    for p in processes:
        p.join()


def run_parallel_mlp(
    rank, world_size,
    input_size, output_size,
    inputs, weights, biases, outputs,
    weight_grads, bias_grads
):
    os.environ['MASTER_ADDR'] = MASTER_ADDR
    os.environ['MASTER_PORT'] = MASTER_PORT
    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=world_size
    )

    torch.use_deterministic_algorithms(True)
    torch.random.manual_seed(rank)

    hidden_size = output_size * 4
    model = nn.Sequential(
        ColumnParallelLinear(input_size, hidden_size),
        nn.ReLU(),
        RowParallelLinear(hidden_size, output_size),
    )

    def load_data(model, layer_idx, idx):
        if layer_idx == 0:
            partition_size = weights[idx].shape[0] // world_size
        elif layer_idx == 2:
            partition_size = weights[idx].shape[1] // world_size

        partition_start, partition_end = rank * partition_size, (rank + 1) * partition_size

        if layer_idx == 0:
            model[layer_idx].weight.data = weights[idx][partition_start: partition_end].detach().requires_grad_(True)
            model[layer_idx].bias.data = biases[idx][partition_start:partition_end].detach().requires_grad_(True)
        elif layer_idx == 2:
            model[layer_idx].weight.data = weights[idx][:, partition_start:partition_end].detach().requires_grad_(True)
            model[layer_idx].bias.data = biases[idx][:partition_end].detach().requires_grad_(True)
        return model

    model = load_data(model, layer_idx=0, idx=0)
    model = load_data(model, layer_idx=2, idx=1)

    outputs_parallel = model(inputs)
    outputs_parallel.sum().backward()

    assert torch.allclose(outputs_parallel, outputs, rtol=0.01)

    for layer_idx, grad_idx in [[0, 0], [2, 1]]:
        if layer_idx == 0:
            partition_size = weight_grads[grad_idx].shape[0] // world_size
            grad_chunks = torch.split(weight_grads[grad_idx], partition_size, dim=0)
            bias_chunks = torch.split(bias_grads[grad_idx], partition_size, dim=0)
        elif layer_idx == 2:
            partition_size = weight_grads[grad_idx].shape[1] // world_size
            grad_chunks = torch.split(weight_grads[grad_idx], partition_size, dim=1)

        assert torch.allclose(model[layer_idx].weight.grad, grad_chunks[rank])
        if layer_idx == 0:
            assert torch.allclose(model[layer_idx].bias.grad, bias_chunks[rank])
        else:
            assert torch.allclose(model[layer_idx].bias.grad, bias_grads[grad_idx])

    dist.destroy_process_group()


def test_parallel_mlp():
    processes = []
    world_size = 4
    batch_size, input_size, output_size = 10, 16, 12
    hidden_size = output_size * 4

    inputs = torch.randn(batch_size, input_size, requires_grad=False)

    model = nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, output_size),
    )
    outputs = model(inputs)
    outputs.sum().backward()

    weights = [
        model[0].weight.data.detach(),
        model[2].weight.data.detach(),
    ]
    biases = [model[0].bias.data.detach(), model[2].bias.data.detach()]
    weight_grads = [
        model[0].weight.grad.detach().requires_grad_(False),
        model[2].weight.grad.detach().requires_grad_(False)
    ]
    bias_grads = [
        model[0].bias.grad.detach().requires_grad_(False),
        model[2].bias.grad.detach().requires_grad_(False)

    ]

    for rank in range(world_size):
        p = Process(target=run_parallel_mlp, args=(
            rank, world_size,
            input_size, output_size,
            # Because PyTorch does not support sending tensors
            # that require gradients through inter-process communication
            # we need to detach them from the computational graph
            inputs, deepcopy(weights), deepcopy(biases), outputs.detach(),
            deepcopy(weight_grads), deepcopy(bias_grads)
        ))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()
