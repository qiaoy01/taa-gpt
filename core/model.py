import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, causal_attention=True, dropout_rate=0.1):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads
        self.causal_attention = causal_attention

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout_rate)

    def split_heads(self, x):
        batch_size, seq_len, _ = x.size()
        return x.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

    def forward(self, x):
        q = self.split_heads(self.W_q(x))
        k = self.split_heads(self.W_k(x))
        v = self.split_heads(self.W_v(x))

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / self.d_k**0.5
        
        if self.causal_attention:
            mask = torch.triu(torch.ones_like(attn_weights), diagonal=1).bool()
            attn_weights = attn_weights.masked_fill(mask, float('-inf'))

        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, v).transpose(1, 2).contiguous()
        attn_output = attn_output.view(*x.shape)
        attn_output = self.linear(attn_output)
        attn_output = self.dropout(attn_output)
        return attn_output
    
class GPTDecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, causal_attention=True, dropout_rate=0.1):
        super(GPTDecoderLayer, self).__init__()
        self.multihead_attention = MultiHeadAttention(d_model = d_model, 
                                                      num_heads = num_heads, 
                                                      causal_attention = causal_attention, 
                                                      dropout_rate = dropout_rate)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_ff, d_model),
            nn.LayerNorm(normalized_shape=d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        attn_output = self.multihead_attention(x)
        x = self.norm1(x + self.dropout(attn_output))
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        return x
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                self.pos_encoding[pos, i] = math.sin(pos / (10000 ** ((2 * i) / d_model)))
                self.pos_encoding[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))
        self.pos_encoding = self.pos_encoding.unsqueeze(0)

    def forward(self, x):
        x = x + self.pos_encoding[:, :x.size(1), :].to(x.device)
        return x
    
class GPTDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, max_seq_len, causal_attention=True, dropout_rate=0.1):
        super(GPTDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        self.layers = nn.ModuleList([GPTDecoderLayer(d_model = d_model, 
                            num_heads = num_heads, 
                            d_ff = d_ff, 
                            causal_attention = True, 
                            dropout_rate = dropout_rate
                            ) for _ in range(num_layers)
                            ])
        self.fc = nn.Linear(d_model, vocab_size)
        self.max_seq_len = max_seq_len
        self.dropout = nn.Dropout(dropout_rate)
        self.layerNorm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.layerNorm(self.dropout(self.pos_encoding(self.embedding(x))))
        for layer in self.layers:
            x = layer(x)
        x = self.fc(x)
        return x
    
    def generate(self, start_tokens, max_length=128, end_token=None, temperature=1.0, top_k=10):
        self.eval()
        with torch.no_grad():
            tokens = start_tokens
            for _ in range(max_length):
                logits = self.forward(tokens)
                next_token_logits = logits[:, -1, :] / temperature

                if top_k is not None:
                    top_k_logits, top_k_indices = next_token_logits.topk(top_k, dim=-1)
                    next_token_logits[top_k_indices != top_k_indices] = -float('Inf')
                next_token_probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(next_token_probs, num_samples=1)
                tokens = torch.cat([tokens, next_token], dim=-1)

                if end_token is not None and next_token.item() == end_token:
                    break

            return tokens



