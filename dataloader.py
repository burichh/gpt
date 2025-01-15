import numpy as np
import os
import torch
from typing import Literal
from ddp_config import DDPConfig

class DataLoader():
    
    def __init__(self, batch_size: int, sequence_length: int, split: Literal["train", "val"], ddp_config: DDPConfig):
        self.B = batch_size
        self.T = sequence_length
        self.shard_index = 0
        self.shard_names = load_shard_names(split)
        self.tokens = load_shard(self.shard_names[self.shard_index])
        self.process_rank = ddp_config.rank
        self.num_processes = ddp_config.world_size
        self.current_position = self.B * self.T * self.process_rank

    def get_next_batch(self):
        
        B, T = self.B, self.T
        buf = torch.tensor(self.tokens[self.current_position : self.current_position + (B * T + 1)])
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)

        self.current_position += B * T * self.num_processes

        if self.current_position + (B * T + 1) > len(self.tokens):
            self.shard_index = (self.shard_index + 1) % len(self.shard_names)
            self.tokens = load_shard(self.shard_names[self.shard_index])
            self.current_position = B * T * self.process_rank

        return x, y
    
def load_shard_names(split):
    shard_names = [shard_name for shard_name in os.listdir("edu_fineweb10B") if split in shard_name]
    full_shard_names = [os.path.join("edu_fineweb10B", shard_name) for shard_name in shard_names]
    return full_shard_names

def load_shard(shard_name):
    tokens = np.load(shard_name)
    tokens = tokens.astype(np.int64)
    return tokens