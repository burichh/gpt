import torch
import torch.nn as nn
from dataloader import DataLoader
import time
import math
from typing import Literal

class CosineDecayLRScheduler:
    def __init__(self, warmup_steps: int, max_step: int, max_learning_rate: float, min_learning_rate: float):
        self.warmup_steps = warmup_steps
        self.max_step = max_step
        self.max_learning_rate = max_learning_rate
        self.min_learning_rate = min_learning_rate

    def calculate(self, step):
        if step < self.warmup_steps:
            return self.max_learning_rate * (step + 1) / self.warmup_steps
        if step > self.max_step:
            return self.min_learning_rate
        decay_progress = (step - self.warmup_steps) / (self.max_step - self.warmup_steps) # in the range of [0, 1]
        decay_coeff = (1 + math.cos(decay_progress * math.pi)) / 2 # cosine going down from 1 to 0
        return self.min_learning_rate + (self.max_learning_rate - self.min_learning_rate) * decay_coeff # cosine going down from max_learning_rate to min_learning_rate

class Trainer:
    def __init__(self,
                 model: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 lr_scheduler: CosineDecayLRScheduler,
                 train_loader: DataLoader,
                 total_batch_size: int,
                 device: Literal["cpu", "cuda"]):
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_loader = train_loader
        self.device = device

        assert total_batch_size % (train_loader.B * train_loader.T) == 0

        self.total_batch_size = total_batch_size
        self.grad_accum_steps = total_batch_size // (train_loader.B * train_loader.T)
        print(f"total batch size: {total_batch_size}")
        print(f"num of gradient accumulation steps: {self.grad_accum_steps}")
    
    def train(self, num_of_steps: int) -> None:
        for step_index in range(num_of_steps):
            t0 = time.time()

            loss, grad_norm, learning_rate = self._training_step(step_index)

            torch.cuda.synchronize()
            t1 = time.time()
            dt = (t1 - t0) # in seconds
            B, T = self.train_loader.B, self.train_loader.T
            tokens_processed = B * T * self.grad_accum_steps
            tokens_per_sec = tokens_processed / dt
            print(f"step: {step_index + 1} | loss: {loss.item()} | lr: {learning_rate:.4e} | grad_norm: {grad_norm:.4f} | dt: {1000*dt:.2f}ms | tokens/sec: {tokens_per_sec:.2f}")
    
    def _training_step(self, step_index: int):
        model, optimizer, grad_accum_steps = self.model, self.optimizer, self.grad_accum_steps

        optimizer.zero_grad()
        total_loss = 0.0

        for microstep in range(grad_accum_steps):
            x, y = self.train_loader.get_next_batch()
            x, y = x.to(self.device), y.to(self.device)
            with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
                logits, loss = model(x, y)
            loss = loss / grad_accum_steps
            total_loss += loss.detach()
            loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        learning_rate = self.lr_scheduler.calculate(step_index)
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate
        optimizer.step()

        return total_loss, grad_norm, learning_rate