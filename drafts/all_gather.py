import os
import torch
import torch.distributed as dist
from torch.multiprocessing import Process

def run(rank, size):
    # Set up distributed environment
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(size)
    dist.init_process_group("gloo", init_method="tcp://127.0.0.1:23456", rank=rank, world_size=size)

    # Create a tensor on each process
    tensor = torch.tensor([rank], dtype=torch.float32)

    # Allocate memory for the gathered tensors
    gathered_tensors = [torch.zeros(1) for _ in range(dist.get_world_size())]

    # Perform all-gather
    dist.all_gather(gathered_tensors, tensor)

    # Print the gathered tensors
    print(f"Rank {rank}: xs = {gathered_tensors}")

    # print(f"rank::::: {torch.distributed.get_rank()}")

def main():
    world_size = 2
    processes = []

    for rank in range(world_size):
        p = Process(target=run, args=(rank, world_size))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

if __name__ == "__main__":
    main()
