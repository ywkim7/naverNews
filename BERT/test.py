import os
import torch.distributed as dist

dist.init_process_group(backend='nccl', init_method='env://')
world_size = dist.get_world_size()
rank = dist.get_rank()
local_rank = int(os.environ['LOCAL_RANK'])

print(local_rank)