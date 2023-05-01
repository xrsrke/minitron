from typing import Callable, Optional, List, Dict, Tuple
from queue import Queue
import sys
from contextlib import contextmanager

import torch

from .stream import use_stream, use_device, AbstractStream

InQueue = Queue
OutQueue = Queue


class Batch:
    pass


class Task:
    def __init__(
        self,
        stream: AbstractStream,
        compute: Callable[[], Batch],
        finalize: Optional[Callable[[Batch], None]]
    ) -> None:
        self.stream = stream
        self._compute = compute
        self._finalize = finalize

    def compute(self) -> Batch:
        with use_stream(self.stream):
            return self._compute()

    def finalize(self, batch: Batch) -> None:
        if self._finalize is None:
            return

        with use_stream(self.stream):
            if self._finalize is not None:
                self._finalize(batch)


def worker(
    in_queue: InQueue,
    out_queue: OutQueue,
    device: torch.device,
    grad_mode: bool
):
    torch.set_grad_enabled(grad_mode)

    with use_device(device):
        while True:
            task = in_queue.get()

            if task is None:
                break

            try:
                batch = task.compute()
            except Exception:
                exec_info = sys.exec_info()
                out_queue.put((False, exec_info))
                continue

            out_queue.put((True, (task, batch)))

        done = (False, None)
        out_queue.put(done)


@contextmanager
def spawn_workers(
    devices: List[torch.device]
):
    in_queues: List[InQueue] = []
    output_queues: List[OutQueue] = []

    workers: Dict[torch.device, Tuple[InQueue, OutQueue]] = {}

    for device in devices:
        try:
            in_queue, out_queue = workers[device]
        except KeyError:
            pass
