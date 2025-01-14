import torch
from dataloader import DataLoader
from model import GPT, GPTConfig
from trainer import Trainer, CosineDecayLRScheduler
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

#from ddp_config import DDPConfig, NoDDP, create_ddp_config

#ddp_config = create_ddp_config()

device = "cuda" if torch.cuda.is_available() else "cpu"
#print(f"using device: {ddp_config.device_type} | num_of_processes: {ddp_config.world_size}")

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

train_loader = DataLoader(batch_size=B, sequence_length=T, split='train')#, ddp_config=ddp_config)
val_loader = DataLoader(batch_size=B, sequence_length=T, split='val')#, ddp_config=ddp_config)
model = GPT(GPTConfig())
model.to(device)
model = torch.compile(model)
optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4)
lr_scheduler = CosineDecayLRScheduler(warmump_steps, max_step, max_learning_rate, min_learning_rate)
trainer = Trainer(model, optimizer, lr_scheduler, train_loader, val_loader, total_batch_size, device, log_dir)

trainer.train(num_of_steps=max_step)

import sys; sys.exit(0)


import tiktoken

num_return_sequences = 5
max_length = 30

model = GPT(GPTConfig())
model.eval()
model.to(device)

import tiktoken
enc = tiktoken.get_encoding('gpt2')
tokens = enc.encode("Hello, I'm a language model,")
tokens = torch.tensor(tokens, dtype=torch.long)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
x = tokens.to(device)

torch.manual_seed(42)
torch.cuda.manual_seed(42)
while x.size(1) < max_length:
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits[:, -1], dim=-1)
        top_k_probs, top_k_indices = torch.topk(probs, 50, dim=-1) # top indices: (B, 50)
        next_token_index = torch.multinomial(top_k_probs, 1) # next_token_index: (B, 1)
        next_token = torch.gather(top_k_indices, 1, next_token_index)
        x = torch.cat((x, next_token), dim=-1)

for i in range(num_return_sequences):
    print(enc.decode(x[i].tolist()))




# import code; code.interact(local=locals())