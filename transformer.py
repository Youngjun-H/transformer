import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SelfAttention(nn.Module):
    def __init__(self, embed_dim, atten_dim):
        super().__init__()
        self.query = nn.Linear(embed_dim, atten_dim)
        self.key = nn.Linear(embed_dim, atten_dim)
        self.value = nn.Linear(embed_dim, atten_dim)

    def forward(self, x):
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        scores = scores/key.size(-1)**0.5

        attn_weights = F.softmax(scores, dim=-1)
        weighted_value = torch.matmul(attn_weights, value)

        return weighted_value

class MaskedSelfAttention(nn.Module):
    def __init__(self, embed_dim, atten_dim):
        super().__init__()
        self.query = nn.Linear(embed_dim, atten_dim, bias=False)
        self.key = nn.Linear(embed_dim, atten_dim, bias=False)
        self.value = nn.Linear(embed_dim, atten_dim, bias=False)
        
    def forward(self, x):
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        
        scores = torch.matmul(query, key.transpose(-2, -1))
        scores = scores / key.size(-1)**0.5
        
        tril = torch.tril(torch.ones(x.size(1), x.size(1))).to(x.device)
        masked_scores = scores.masked_fill(tril==0, float('-inf'))
        
        attention_weights = F.softmax(masked_scores, dim=-1)
        weighted_values = torch.matmul(attention_weights, value)

        return weighted_values
    
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        atten_dim = embed_dim // num_heads
        self.attentions = nn.ModuleList([SelfAttention(embed_dim, atten_dim) for _ in range(num_heads)])
        self.fc = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, x):
        head_outputs = []
        for attention in self.attentions:
            head_outputs.append(attention(x))
        concatenated_heads = torch.cat(head_outputs, dim=-1)
        return self.fc(concatenated_heads)
    
class FeedForward(nn.Module):
    def __init__(self, embed_dim, ff_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
    
    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.multihead_attn = MultiHeadAttention(embed_dim, num_heads)
        
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.ff = FeedForward(embed_dim, ff_dim)        
    
    def forward(self, x):
        x = x + self.multihead_attn(self.layer_norm1(x))
        x = x + self.ff(self.layer_norm2(x))
        return x
    

