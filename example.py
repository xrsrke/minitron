import os
import torch
import torch.distributed as dist
from torch.multiprocessing import Process

def distributed_sum_of_squares(rank, size):
    tensor = torch.tensor(rank, dtype=torch.float) ** 2
    sum_tensor = torch.zeros(1, dtype=torch.float)

    dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=dist.group.WORLD)
    dist.all_reduce(sum_tensor, op=dist.ReduceOp.SUM, group=dist.group.WORLD)

    sum_tensor += tensor

    print(f"Rank {rank}: tensor={tensor.item()}, sum_tensor={sum_tensor.item()}")

def run(rank, size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    dist.init_process_group("gloo", rank=rank, world_size=size)

    distributed_sum_of_squares(rank, size)

def main():
    world_size = 4
    processes = []

    for rank in range(world_size):
        p = Process(target=run, args=(rank, world_size))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

if __name__ == "__main__":
    main()