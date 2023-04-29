# import torch
# import torch.distributed as dist
# import os
# from torch.multiprocessing import Process

# # Define the function that will be run by each process
# def run_process(rank, size):
#     # Set up the distributed process group
#     dist.init_process_group("gloo", rank=rank, world_size=size)

#     # Create a tensor with the rank of the process
#     tensor = torch.tensor(float(rank)).to(torch.float32)
#     print(f"Rank {rank}: Initial tensor value: {tensor}")

#     # Perform the all_reduce operation
#     dist.all_reduce(tensor)
#     print(f"Rank {rank}: Tensor value after all_reduce: {tensor}")

#     # Clean up
#     dist.destroy_process_group()

# def main():
#     # Set up the distributed environment
#     world_size = 4
#     os.environ["MASTER_ADDR"] = "localhost"
#     os.environ["MASTER_PORT"] = "29500"

#     # Create a list of processes
#     processes = []
#     for rank in range(world_size):
#         p = Process(target=run_process, args=(rank, world_size))
#         p.start()
#         processes.append(p)

#     # Wait for all processes to finish
#     for p in processes:
#         p.join()

# if __name__ == "__main__":
#     main()


####################################################################################################


# import torch
# import torch.distributed as dist
# import os
# import sys
# from torch.multiprocessing import Process

# # def get_model_parallel_group():
# #     return dist.group.WORLD

# def gather_and_concat(input_):
#     world_size = dist.get_world_size()

#     if world_size == 1:
#         return input_

#     dt = input_.dtype

#     last_dim = input_.dim() - 1
#     rank = dist.get_rank()

#     tensor_list = [torch.empty_like(input_) for _ in range(world_size)]
#     tensor_list[rank] = input_
#     dist.all_gather(tensor_list, input_)

#     output = torch.cat(tensor_list, dim=last_dim).contiguous()

#     return output

# def run(rank, size):
#     os.environ['MASTER_ADDR'] = 'localhost'
#     os.environ['MASTER_PORT'] = '29500'
#     dist.init_process_group("gloo", rank=rank, world_size=size)

#     input_ = torch.rand(2, 2).to(rank)
#     print(f"Rank {rank}: Original tensor: {input_}")

#     output = gather_and_concat(input_)
#     print(f"Rank {rank}: Gathered and concatenated tensor: {output}")

#     dist.destroy_process_group()

# def main():
#     world_size = 2
#     processes = []

#     for rank in range(world_size):
#         p = Process(target=run, args=(rank, world_size))
#         p.start()
#         processes.append(p)

#     for p in processes:
#         p.join()

# if __name__ == "__main__":
#     main()

