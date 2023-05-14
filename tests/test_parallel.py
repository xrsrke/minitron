from minitron.mpu.parallel import ParallelState


def test_parallel_state():
    INTRA_LAYER_PARALLEL_SIZE = 2
    INTER_LAYER_PARALLEL_SIZE = 2
    DATA_PARALLEL_SIZE = 2
    WORLD_SIZE = 4

    state = ParallelState(
        intra_layer_parallel_size=INTRA_LAYER_PARALLEL_SIZE,
        inter_layer_parallel_size=INTER_LAYER_PARALLEL_SIZE,
        data_parallel_size=DATA_PARALLEL_SIZE,
        world_size=WORLD_SIZE
    )

    assert state.is_init is True
    assert state.world_size == WORLD_SIZE
    assert state.intra_layer_parallel_size == INTRA_LAYER_PARALLEL_SIZE
    assert state.inter_layer_parallel_size == INTER_LAYER_PARALLEL_SIZE
    assert state.data_parallel_size == DATA_PARALLEL_SIZE
