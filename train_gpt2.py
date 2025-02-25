from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int
    n_layer: int
    n_head: int
    n_embd: int
    dropout: float = 0.0
class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)
    def forward(self, idx, targets=None): # idx and targets are both (B,T) tensor
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.sa = nn.ModuleList([
            nn.Linear(config.n_embd, config.n_embd) for _ in range(3)
        ])
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4*config.n_embd),
            nn.ReLU(),
            nn.Linear(4*config.n_embd, config.n_embd),
        )
        self.ln2 = nn.LayerNorm(config.n_embd)
    def forward(self, x):
        x = x + self.sa[0](x)
        x = self.ln1(x)
        x = x + self.sa[1](x)
        x = self.ln2(x)
        x = x + self.mlp(x)
        return x
if __name__ == '__main__':  
    config = GPTConfig(vocab_size=50304, block_size=1024, n_layer=12, n_head=12, n_embd=768)
    model = GPT(config)
    idx = torch.randint(0, 50304, (2, 1024))
    logits, loss = model(idx)
    print(logits.shape)  # (2, 1024, 50304)
    print(loss)  # tensor(9.2103)
    logits, loss = model(idx, targets=idx)
    print(logits.shape)  # (2, 1024, 50304)
    print(loss)  # tensor(9.2103)
    logits, loss = model(idx, targets=idx[:, 1:])
    print(logits.shape)  # (2, 1024, 50304)
    print(loss)  # tensor(9.2103)
    logits, loss = model(idx, targets=idx[:, :-1])
    print(logits.shape)  # (2, 1024, 50304)
    print(loss)  # tensor(9.2103)
    logits, loss = model(idx, targets=idx[:, 1:-1])
    print(logits.shape)  # (2, 1024, 50304)