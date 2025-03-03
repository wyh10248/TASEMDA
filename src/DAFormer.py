import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class RelativePositionEncoding(nn.Module):
    def __init__(self, num_heads, max_len):
        super().__init__()
        self.num_heads = num_heads
        self.relative_position_embeddings = nn.Parameter(
            torch.randn(max_len, max_len, num_heads)
        )

    def forward(self, seq_len):
        return self.relative_position_embeddings[:seq_len, :seq_len, :]

class SoftMask(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.gate = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x):
        mask = torch.sigmoid(self.gate(x))
        return x * mask

class BiasedMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, bias_factor=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.bias_factor = bias_factor

    def forward(self, x, attn_mask=None, relative_pos=None):
        batch_size, seq_len, _ = x.shape
        
        # Project queries, keys, values
        q = self.q_proj(x)  # (B, T, E)
        k = self.k_proj(x)  # (B, T, E)
        v = self.v_proj(x)  # (B, T, E)
        
        # Reshape to multi-head
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, T, D)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)  # (B, H, T, T)
        
        # Add relative position bias
        if relative_pos is not None:
            # relative_pos: (T, T, H) -> (H, T, T)
            relative_pos = relative_pos.permute(2, 0, 1).contiguous()
            attn_scores += self.bias_factor * relative_pos.unsqueeze(0)  # Add per-head bias
        
        # Apply attention mask
        if attn_mask is not None:
            attn_scores += attn_mask.unsqueeze(0).unsqueeze(0)
        
        # Compute attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)  # (B, H, T, D)
        
        # Merge heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)
        
        return attn_output

class DomainAdaptedTransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, dim_feedforward=2048, dropout=0.1, max_len=1931):
        super().__init__()
        self.relative_pos_encoding = RelativePositionEncoding(num_heads, max_len)
        self.self_attn = BiasedMultiheadAttention(embed_dim, num_heads)
        self.soft_mask = SoftMask(embed_dim)

        self.linear1 = nn.Linear(embed_dim, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, embed_dim)
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        seq_len = x.shape[1]
        relative_pos = self.relative_pos_encoding(seq_len)
        
        attn_output = self.self_attn(x, relative_pos=relative_pos)
        attn_output = self.dropout(attn_output)
        x = self.norm1(x + attn_output)
        
        x = self.soft_mask(x)

        ffn_output = self.linear2(F.relu(self.linear1(x)))
        ffn_output = self.dropout(ffn_output)
        x = self.norm2(x + ffn_output)
        return x

class DomainAdaptedTransformerEncoder(nn.Module):
    def __init__(self, num_layers, embed_dim, num_heads, max_len=500):
        super().__init__()
        self.layers = nn.ModuleList([
            DomainAdaptedTransformerEncoderLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dim_feedforward=512,
                dropout=0.1,
                max_len=max_len
            )
            for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            out = x.squeeze(0)
        return out

#-------------------------------------------------
if __name__ == '__main__':
    model = DomainAdaptedTransformerEncoder(num_layers=2, embed_dim=50, num_heads=10, max_len=500)
    inputx = torch.randn(1, 50, 50)  # (batch_size, seq_len, embed_dim)
    O = model(inputx)
    print(O.shape)