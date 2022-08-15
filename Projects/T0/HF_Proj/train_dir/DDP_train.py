import os
import torch
from argparse import ArgumentParser

os.environ["RANK"] = "0"
os.environ["LOCAL_RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "8080"
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3, 4, 5, 6"

torch.distributed.init_process_group(backend="nccl")  # 并行训练初始化，建议'nccl'模式

print(torch.cuda.device_count())  # 打印gpu数量
print('world_size', torch.distributed.get_world_size())  # 打印当前进程数

parser = ArgumentParser(description='Pytorch ...')
parser.add_argument('--local_rank', default=-1, type=int)
args = parser.parse_args()
torch.cuda.set_device(args.local_rank)  # 设置gpu编号为local_rank;此句也可能看出local_rank的值是什么
print(torch.cuda.current_device())