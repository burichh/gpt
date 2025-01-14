import torch
import time
import math
import os
import torch.nn.functional as F
import tiktoken
from typing import Literal
from dataloader import DataLoader
from model import GPT
from hellaswag import iterate_examples, render_example

class CosineDecayLRScheduler:
    def __init__(self, warmup_steps: int, max_step: int, max_learning_rate: float, min_learning_rate: float) -> None:
        self.warmup_steps = warmup_steps
        self.max_step = max_step
        self.max_learning_rate = max_learning_rate
        self.min_learning_rate = min_learning_rate

    def calculate(self, step: int) -> float:
        if step < self.warmup_steps:
            return self.max_learning_rate * (step + 1) / self.warmup_steps
        if step > self.max_step:
            return self.min_learning_rate
        decay_progress = (step - self.warmup_steps) / (self.max_step - self.warmup_steps) # in the range of [0, 1]
        decay_coeff = (1 + math.cos(decay_progress * math.pi)) / 2 # cosine going down from 1 to 0
        return self.min_learning_rate + (self.max_learning_rate - self.min_learning_rate) * decay_coeff # cosine going down from max_learning_rate to min_learning_rate

class Trainer:
    def __init__(self,
                 model: GPT,
                 optimizer: torch.optim.Optimizer,
                 lr_scheduler: CosineDecayLRScheduler,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 total_batch_size: int,
                 device: Literal["cpu", "cuda"],
                 log_dir: str,
                 ddp: bool = False) -> None:
        
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, f"log.txt")
        with open(self.log_file, "w") as f: # open for writing to clear the file
            pass

        assert total_batch_size % (train_loader.B * train_loader.T) == 0

        self.total_batch_size = total_batch_size
        self.grad_accum_steps = total_batch_size // (train_loader.B * train_loader.T)
        print(f"total batch size: {total_batch_size}")
        print(f"num of gradient accumulation steps: {self.grad_accum_steps}")
    
    def train(self, num_of_steps: int) -> None:

        for step in range(num_of_steps):
            last_step = (step == (num_of_steps - 1))

            if step % 250 == 0 or last_step:
                val_loss = self._validation_step(step)
                self._evaluate_hellaswag(step)
                self._generate_and_print_text("Hello I'm a language model,", length=30, num_of_generated_sequences=5)

                if step > 0 and (step % 5000 == 0 or last_step):
                    self._save_checkpoint(step, val_loss)

            self._training_step(step)

    def _training_step(self, step: int) -> None:
        model, optimizer, grad_accum_steps = self.model, self.optimizer, self.grad_accum_steps

        self.model.train()
        optimizer.zero_grad()
        total_loss = 0.0
        t0 = time.time()

        for microstep in range(grad_accum_steps):
            x, y = self.train_loader.get_next_batch()
            x, y = x.to(self.device), y.to(self.device)
            with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
                logits, loss = model(x, y)
            loss = loss / grad_accum_steps
            total_loss += loss.detach()
            loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        learning_rate = self.lr_scheduler.calculate(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate
        optimizer.step()

        torch.cuda.synchronize()
        t1 = time.time()

        self._log_training_step(step, total_loss, learning_rate, grad_norm, (t1 - t0))

    def _log_training_step(self, step: int, loss: float, learning_rate: float, grad_norm: float, dt: float):
        B, T = self.train_loader.B, self.train_loader.T
        tokens_processed = B * T * self.grad_accum_steps
        tokens_per_sec = tokens_processed / dt
        print(f"step: {step + 1} | loss: {loss:.6f} | lr: {learning_rate:.4e} | grad_norm: {grad_norm:.4f} | dt: {1000*dt:.1f}ms | tokens/sec: {tokens_per_sec:.1f}")

        with open(self.log_file, 'a') as f:
            f.write(f"{step} train {loss:.6f}\n")
    
    @torch.no_grad()
    def _validation_step(self, step):
        self.model.eval()
        total_loss = 0.0
        val_steps = 20
        for microstep in range(val_steps):
            x, y = self.val_loader.get_next_batch()
            x, y = x.to(self.device), y.to(self.device)
            with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
                logits, loss = self.model(x, y)
            loss = loss / val_steps
            total_loss += loss.detach()

        print(f"validation loss: {total_loss.item():.4f}")
        with open(self.log_file, 'a') as f:
            f.write(f"{step} val {total_loss.item():.4f}\n")

        return total_loss

    @torch.no_grad()
    def _evaluate_hellaswag(self, step) -> None:
        self.model.eval()
        num_correct = 0
        num_total = 0
        for example in iterate_examples("val"):
            data, tokens, mask, label = render_example(example)
            tokens = tokens.to(self.device)
            mask = mask.to(self.device)
            with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
                logits, loss = self.model(tokens)
            predicted_answer = get_most_likely_answer(logits, tokens, mask)

            num_total += 1
            num_correct += int(predicted_answer == label)
            accuracy = num_correct / num_total

        print(f"Hellaswag accuracy: {num_correct} / {num_total} = {accuracy:.4f}")
        with open(self.log_file, 'a') as f:
            f.write(f"{step} hella {accuracy:.4f}\n")


    @torch.no_grad()
    def _save_checkpoint(self, step: int, val_loss: torch.Tensor) -> None:
        checkpoint_path = os.path.join(self.log_dir, f"model_{step:05d}.pt")
        checkpoint = {
            "model_state_dict" : self.model.state_dict(),
            "model_config" : self.model.config,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "step": step,
            "val_loss": val_loss,
        }
        torch.save(checkpoint, checkpoint_path)

    @torch.no_grad()
    def _generate_and_print_text(self, context: str, length: int, num_of_generated_sequences: int) -> None:
        sample_rng = torch.Generator(device=self.device)
        sample_rng.manual_seed(42)
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(context)
        tokens = torch.tensor(tokens, dtype=torch.long)
        tokens = tokens.unsqueeze(0).repeat(num_of_generated_sequences, 1) # (B, T) input
        x = tokens.to(self.device)

        while x.size(1) < length:
            with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
                logits, loss = self.model(x)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            top_k_probs, top_k_indices = torch.topk(probs, 50, dim=-1) # (B, 50)
            next_token_index = torch.multinomial(top_k_probs, num_samples=1, generator=sample_rng)
            next_token = torch.gather(top_k_indices, dim=-1, index=next_token_index)
            x = torch.cat([x, next_token], dim=-1)

        for i in range(x.size(0)):
            print(enc.decode(x[i].tolist()))
        

def get_most_likely_answer(logits: torch.Tensor, tokens: torch.Tensor, mask: torch.Tensor):
    '''
    logits: (B, T, V) output of the transformer
    tokens: (B, T) target tokens that should have high probability
    mask: (B, T) masking out the context and padding at the end of each answer.

    returns: index of most likely answer.
    '''

    # The last column of logits (at position T-1) are not needed, as they would be only useful to predict a new token at position T,
    # but for hellaswag eval we do not generate new tokens.
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous() # To align the tokens with shift_logits, drop the first token, as it's not needed for loss evaluation anyway.
    shift_mask = (mask[..., 1:]).contiguous() # To align the mask with shift_tokens, drop the first column of the mask.
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
    shift_losses = shift_losses.view(tokens.size(0), -1)
    
    masked_shift_losses = shift_losses * shift_mask # Mask out all the loss values that do not correspond to the answer tokens.

    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)

    # Now we have a loss for each of the 4 completions,
    # the one with the lowest loss should be the most likely.
    return avg_loss.argmin().item()
