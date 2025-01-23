from dataclasses import dataclass
import torch
import os

def is_ddp_enabled() -> bool:
    return int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?

@dataclass
class DDPConfig():
    rank: int
    local_rank: int
    world_size: int
    device: str
    device_type: str
    master_process: bool

def create_ddp_config() -> DDPConfig:
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    device_type = "cuda"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.

    return DDPConfig(ddp_rank, ddp_local_rank, ddp_world_size, device, device_type, master_process)

NO_DDP = DDPConfig(
    rank = 0,
    local_rank = 0,
    world_size = 1,
    device = "cuda" if torch.cuda.is_available() else "cpu",
    device_type = "cuda" if torch.cuda.is_available() else "cpu",
    master_process = True,
)