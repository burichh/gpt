import torch
from dataloader import DataLoader
from model import GPT, GPTConfig
from trainer import Trainer, CosineDecayLRScheduler
from torch.distributed import init_process_group, destroy_process_group

from ddp_config import is_ddp_enabled, create_ddp_config, NO_DDP

if is_ddp_enabled():
    init_process_group(backend='nccl')
    ddp_config = create_ddp_config()
else:
    ddp_config = NO_DDP

print(f"using device: {ddp_config.device_type} | ddp: {is_ddp_enabled()} | num_of_processes: {ddp_config.world_size}")

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

torch.set_float32_matmul_precision('high')

max_step = 19073
warmump_steps = 715
max_learning_rate = 6e-4
min_learning_rate = max_learning_rate * 0.1
total_batch_size = 524288 # from GPT-3 paper (GPT-3 Small)
B = 8
T = 1024
log_dir = "log"

train_loader = DataLoader(batch_size=B, sequence_length=T, split='train', ddp_config=ddp_config)
val_loader = DataLoader(batch_size=B, sequence_length=T, split='val', ddp_config=ddp_config)
model = GPT(GPTConfig())
model.to(ddp_config.device)
if is_ddp_enabled() == False: # Only use torch.compile when DDP is not used (since when DDP is used, hellaswag eval does not work porperly with torch.compile).
    model = torch.compile(model)
optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4)
lr_scheduler = CosineDecayLRScheduler(warmump_steps, max_step, max_learning_rate, min_learning_rate)

trainer = Trainer(model, optimizer, lr_scheduler, train_loader, val_loader, total_batch_size, ddp_config, log_dir)
trainer.train(num_of_steps=max_step)

if is_ddp_enabled():
    destroy_process_group()