import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout, attention_type, window_size):
        super().__init__()
        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.attention_type = attention_type
        self.window_size = window_size

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # must split back into individual heads to compute attention scores
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        if self.attention_type == "local":
            attn_output = self._local_attention(q, k, v, mask)
        else:
            attn_output = self._global_attention(q, k, v, mask)

        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.embed_dim)
        )  # reshape and concatenate attention outputs from all heads

        return self.out_proj(attn_output)

    def _global_attention(self, q, k, v, mask=None):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)

        return attn_output

    def _local_attention(self, q, k, v, mask=None):
        _, _, seq_len, _ = q.shape
        half_window = self.window_size // 2

        # create local attention mask
        local_mask = torch.zeros(seq_len, seq_len, device=q.device)
        for i in range(seq_len):
            start = max(0, i - half_window)
            end = min(seq_len, i + half_window + 1)
            local_mask[i, start:end] = 1

        # combine local and causal attention masks
        if mask is not None:
            local_mask = local_mask.unsqueeze(0).unsqueeze(0)
            combined_mask = local_mask * mask
        else:
            combined_mask = local_mask.unsqueeze(0).unsqueeze(0)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(combined_mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)

        return attn_output


class FeedForward(nn.Module):
    def __init__(self, embed_dim, ff_dim, dropout):
        super().__init__()

        self.linear1 = nn.Linear(embed_dim, ff_dim)
        self.linear2 = nn.Linear(ff_dim, embed_dim)
        self.gelu = nn.GELU(approximate="tanh")
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(self.gelu(self.linear1(x))))


class TransformerBlock(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        ff_dim,
        dropout,
        attention_type,
        window_size,
    ):
        super().__init__()

        self.attention = MultiHeadAttention(
            embed_dim, num_heads, dropout, attention_type, window_size
        )
        self.feed_forward = FeedForward(embed_dim, ff_dim, dropout)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_output = self.attention(self.ln1(x), mask)
        x = x + self.dropout(attn_output)

        ff_output = self.feed_forward(self.ln2(x))
        x = x + self.dropout(ff_output)

        return x


class GPT2(nn.Module):
    def __init__(
        self,
        vocab_size=50257,
        context_len=256,
        window_size=128,
        embed_dim=768,
        num_heads=12,
        num_layers=12,
        ff_dim=3072,
        dropout=0.1,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.context_len = context_len
        self.embed_dim = embed_dim

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(context_len, embed_dim)

        self.transformer_blocks = nn.ModuleList()
        for i in range(num_layers):
            # alternating global and local attention layers
            # even layers -> global attention
            # odd layers -> local attention
            attention_type = "global" if i % 2 == 0 else "local"
            self.transformer_blocks.append(
                TransformerBlock(
                    embed_dim, num_heads, ff_dim, dropout, attention_type, window_size
                )
            )

        self.ln_f = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)
        self.dropout = nn.Dropout(dropout)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, input_ids):
        device = input_ids.device
        _, seq_len = input_ids.size()

        if seq_len > self.context_len:
            raise ValueError(
                f"Input sequence length {seq_len} exceeds maximum context length {self.context_len}"
            )

        position_ids = torch.arange(0, seq_len, dtype=torch.long, device=device)

        token_embeds = self.token_embedding(input_ids)
        position_embeds = self.position_embedding(position_ids)

        x = self.dropout(token_embeds + position_embeds)
        # position embeddings will be broadcast across batch size

        causal_mask = (
            torch.tril(torch.ones(seq_len, seq_len, device=device))
            .unsqueeze(0)
            .unsqueeze(0)
        )  # will be broadcast across batch size and attention heads
        for block in self.transformer_blocks:
            x = block(x, causal_mask)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        return logits

    def generate(
        self, input_ids, max_new_tokens=100, temperature=1.0, top_k=None, top_p=None
    ):
        self.eval()

        with torch.no_grad():
            for _ in range(max_new_tokens):
                logits = self.forward(input_ids)
                next_token_logits = logits[:, -1, :] / temperature

                if top_k is not None:
                    next_token_logits = self._top_k_filtering(next_token_logits, top_k)

                if top_p is not None:
                    next_token_logits = self._top_p_filtering(next_token_logits, top_p)

                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                input_ids = torch.cat([input_ids, next_token], dim=1)

                if input_ids.size(1) >= self.context_len:
                    break

        return input_ids

    def _top_k_filtering(self, logits, top_k):
        top_k_values, _ = torch.topk(logits, top_k)
        logits[logits < top_k_values[:, [-1]]] = -float("inf")

        return logits

    def _top_p_filtering(self, logits, top_p):
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
        sorted_indices_to_remove[:, 0] = False

        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove
        )
        logits[indices_to_remove] = -float("inf")

        return logits
