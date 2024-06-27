import torch
import torch.nn as nn
from torch.nn import functional as F

block_size = 8
batch_size = 32
n_embd = 10  # not used right now because bigram model simpply counting
max_itr = 100
learning_rate = 1e-3
eval_interval = 10
eval_iters = 5

with open('zerotohero/tinyshakespeare.txt', 'r', encoding='utf-8') as f:
    texts = f.read()

chars = sorted(list(set(texts)))
vocab_size = len(chars)
stoi = { ch : i for i, ch in enumerate(chars)}
itos = { i : ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[e] for e in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(texts), dtype = torch.long)
n = int(0.9 * len(data))
train = data[:n]
eval = data[n:]

def get_batch(split):
    data = train if split == 'train' else eval
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix]) # B, T
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]) # B, T
    return x, y

torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Bigram(nn.Module):
    def __init__(self):
        super().__init__()
        self.embd = torch.nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, target=None):
        logits = self.embd(idx)

        if target is None:
            loss = None
        
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            target = target.view(B*T)
            loss = F.cross_entropy(logits, target)

        return logits, loss

    def generate(self, idx, max_new):
        for _ in range(max_new):
            logits = self.embd(idx) # (B,T) -> (B,T,C)
            # generate only for the last time step
            logits = logits[:, -1, :] # becomes B C
            probs = F.softmax(logits, dim=-1) # B C
            idx_next  = torch.multinomial(probs, 1, replacement=True)
            idx = torch.cat((idx, idx_next), dim=1) # B, T+1, C
        return idx
    
model = Bigram()

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for step in range(max_itr):
    # every once in a while evaluate the loss on train and val sets
    if step % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    x, y = get_batch('train')
        
    logits, loss = model(x, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate
context = torch.zeros((1,1), dtype=torch.long)
print(decode(model.generate(context, max_new=100)[0].tolist()))