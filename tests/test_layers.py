import torch

from minitron.mpu.layers import ColumnParallelLinear


def test_column_parallel_linear(parallel_state):
    INPUT_SIZE = 16
    OUTPUT_SIZE = 12

    input = torch.randn(INPUT_SIZE, OUTPUT_SIZE)
    linear = ColumnParallelLinear(INPUT_SIZE, OUTPUT_SIZE)

    output = linear(input)

    assert output.shape == (INPUT_SIZE, OUTPUT_SIZE)
