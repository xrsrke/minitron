import pytest

from minitron.mpu.parallel import ParallelState


@pytest.fixture
def parallel_state():
    INTRA_LAYER_PARALLEL_SIZE = 2
    INTER_LAYER_PARALLEL_SIZE = 2
    DATA_PARALLEL_SIZE = 2
    WORLD_SIZE = 4

    return ParallelState(
        INTRA_LAYER_PARALLEL_SIZE,
        INTER_LAYER_PARALLEL_SIZE,
        DATA_PARALLEL_SIZE,
        WORLD_SIZE
    )
