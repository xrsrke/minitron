class ParallelState:
    """Allocate GPUs resources for pipeline parallelism."""
    def __init__(
        self,
        intra_layer_parallel_size: int,
        inter_layer_parallel_size: int,
        data_parallel_size: int,
        world_size: int
    ) -> None:
        """Initialize parallel state for pipeline parallelism.

        Args:
            intra_layer_parallel_size (int): The number of GPUs that
            responsible for handling the computation of a single layer.
            inter_layer_parallel_size (int): The number of GPUs that
            repsonsible for handle a segment of the model
            data_parallel_size (int): The number of GPUs that a input tensor
            will be split across
            world_size (int): The number of GPUs that will be used for training
        """
        # TODO: implement this
        self._is_init: bool = True
        self._intra_layer_parallel_size: int = intra_layer_parallel_size
        self._inter_layer_parallel_size: int = inter_layer_parallel_size
        self._data_parallel_size: int = data_parallel_size
        self._world_size: int = world_size

        # self._init_parallelization()

    @property
    def is_init(self) -> bool:
        """Check if all GPUs resources are allocated.

        Returns:
            bool: True if yes, False if not.
        """
        return self._is_init

    @property
    def world_size(self) -> int:
        """Get the number of GPUs that will be used for training.

        Returns:
            int: The number of GPUs that will be used for training.
        """
        return self._world_size

    # def _init_parallelization(self):
    #     # TODO: implement initialize model parallel
    #     self._is_init = True
