from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

'''
torch.manual_seed(1337)
device = "cuda" if torch.cuda.is_available() else "cpu"
n_layers = 48
d_model = 1536 
n_heads = 16
d_head = 96 # d_head = d_model / n_heads
n_ctx = 1024
p_dropout = 0.2
max_iter = 5000
batch_size = 32
vocab_size = 27
'''

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

class CausalSelfAttention(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.c_attn = nn.Linear(config.n_embd, config.n_embd * 3)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.SCALE_DOWN = True
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).chunk(3, dim=-1)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # att = q @ k.transpose(-2, -1) / math.sqrt(k.size(-1))
        # att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        # att = F.softmax(att, dim=-1)
        # y = att @ v

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        y = y.transpose(1, 2).reshape(B, T, C)
        y = self.c_proj(y)
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.n_embd * 4)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(config.n_embd * 4, config.n_embd)
        self.c_proj.SCALE_DOWN = True

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd)
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "SCALE_DOWN"):
                std *= (2 * self.config.n_layer) ** -0.5 
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, y=None):
        # idx: B, T
        B, T = idx.shape
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is {self.config.block_size}"
        pos_embd = self.transformer.wpe(torch.arange(T, device=idx.device))
        tok_embd = self.transformer.wte(idx)
        x = tok_embd + pos_embd
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if y is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        config_args = {
            'gpt2' : dict(n_layer=12, n_head=12, n_embd=768),
            'gpt2-medium' : dict(n_layer=24, n_head=16, n_embd=1024),
            'gpt2-large' : dict(n_layer=36, n_head=20, n_embd=1280),
            'gpt2-xl' : dict(n_layer=48, n_head=25, n_embd=1600),
        }[model_type]
        config_args["block_size"] = 1024
        config_args["vocab_size"] = 50257
        config = GPTConfig(**config_args)
        model = cls(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]

        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

        assert len(sd_keys) == len(sd_keys_hf), f"mismatch keys: {len(sd_keys)} != {len(sd_keys_hf)}"

        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

# --------------------------------------------------------

import tiktoken

class DataLoaderLite():
    
    def __init__(self, B, T):
        with open("input.txt", "r") as f:
            text = f.read()
        enc = tiktoken.get_encoding('gpt2')
        self.tokens = enc.encode(text)
        self.current_position = 0
        self.B = B
        self.T = T
        print(f"num of tokens: {len(self.tokens)}")
        print(f"1 epoch = {len(self.tokens) // (B * T)} batches")

    def get_next_batch(self):

        B, T = self.B, self.T
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_position = 0

        buf = torch.tensor(self.tokens[self.current_position : self.current_position + (B * T + 1)])
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)
        self.current_position += B * T
        return x, y

# --------------------------------------------------------
import time

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"using device: {device}")

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

torch.set_float32_matmul_precision('high')

train_loader = DataLoaderLite(B=8, T=1024)
model = GPT(GPTConfig())
model.to(device)
model = torch.compile(model)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

for i in range(50):
    t0 = time.time()
    x, y = train_loader.get_next_batch()
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        logits, loss = model(x, y)
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    t1 = time.time()
    dt = (t1 - t0) * 1000
    tokens_per_sec = (train_loader.B * train_loader.T) / (t1 - t0)
    print(f"step: {i}, loss: {loss.item()}, dt: {dt:.2f}ms, tokens/sec: {tokens_per_sec:.2f}")

import sys; sys.exit(0)

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






'''
class MultiHeadSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.query = nn.Linear(d_model, n_heads * d_head, bias=False) # (C, H*D)
        self.key  = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.value = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(n_ctx, n_ctx)))
        self.proj = nn.Linear(n_heads * d_head, d_model)
        self.dropout = nn.Dropout(p_dropout)

    def __call__(self, x):
        B, T, C = x.shape
        Q = self.query(x).view(B, T, n_heads, d_head).transpose(1, 2)  # (B, T, H*D) --> (B, T, H, D) --> (B, H, T, D)
        K = self.key(x).view(B, T, n_heads, d_head).transpose(1, 2)
        attention = Q @ K.transpose(-1, -2) / d_head**0.5 # (B, H, T, D) @ (B, H, D, T) --> (B, H, T, T)
        attention = attention.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        attention = F.softmax(attention, dim=-1)
        V = self.value(x).view(B, T, n_heads, d_head).transpose(1, 2)
        out = attention @ V # (B, H, T, T) @ (B, H, T, D) --> (B, H, T, D) 
        out = out.transpose(1, 2).reshape(B, T, C)
        out = self.proj(out)

        return out

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, 4*d_model),
            nn.ReLU(),
            nn.Linear(4*d_model, d_model),
            nn.Dropout(p_dropout)
        )
        self.multi_head_attention = MultiHeadSelfAttention()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        
    def __call__(self, x):
        x = x + self.multi_head_attention(self.ln1(x))
        x = x + self.feedforward(self.ln2(x))
        return x

class LanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_embedding = nn.Embedding(n_ctx, d_model)
        self.attention_blocks = nn.Sequential(*[Block() for _ in range(n_layer)])
        self.ln_head = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)

        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


    def __call__(self, idx, y=None):
        B, T = idx.shape
        token_embedding = self.token_embedding(idx)
        positional_embedding = self.positional_embedding(torch.arange(0, T, device=device))
        x = token_embedding + positional_embedding
        x = self.attention_blocks(x)
        logits = self.head(self.ln_head(x))
        
        if y is not None:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            y = y.view(B * T)
            loss = F.cross_entropy(logits, y)
        else:
            loss = None
        
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, n_samples):
        for _ in range(n_samples):
            idx_in_block = idx[:, -block_size:]
            logits, _ = self(idx_in_block)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=1) # probs shape (1, V)
            next_idx = torch.multinomial(probs, 1)
            idx = torch.cat((idx, next_idx), dim=1)
        return idx

# SECTION 1:
# check huggingface gpt2
# training loop with AdamW
# periodically sample results
# fast attention
# data loader

# SECTION 2:
# GPU, mixed precision, torch compile, flash attention

# SECTION 3:
# AdamW, hyperparams, gradient clipping, lr scheduler distributed data parallel, eval (HellaSwag)


# Create dataset
def create_dataset(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    chars = sorted(list(set(text)))

    stoi = {s:i for i, s in enumerate(chars)}
    itos = {i:s for s, i in stoi.items()}
    vocab_size = len(chars)

    encode = lambda t : [stoi[char] for char in t]
    decode = lambda t : ''.join([itos[idx] for idx in t])

    train_data = text[:int(0.9 * len(text))]
    val_data = text[int(0.9 * len(text)):]

    Xtr = torch.tensor(encode(train_data))
    Xval = torch.tensor(encode(val_data))

    return Xtr, Xval, chars

def get_batch(split="train"):
    X = Xtr if split == "train" else Xval
    idx = torch.randint(len(X) - block_size, (batch_size,))
    Xb = torch.stack([X[i:i+block_size] for i in idx])
    Yb = torch.stack([X[i+1:i+block_size+1] for i in idx])
    Xb, Yb = Xb.to(device), Yb.to(device)
    return Xb, Yb
'''