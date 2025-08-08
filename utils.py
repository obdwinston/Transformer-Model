import torch
from torch.utils.data import DataLoader, Dataset
from collections import Counter


class TextDataset(Dataset):
    def __init__(self, text, tokenizer, context_len, token_mapping):
        self.tokens = get_encoding(text, tokenizer, token_mapping)
        self.context_len = context_len

    def __len__(self):
        return len(self.tokens) // self.context_len

    def __getitem__(self, idx):
        start_idx = idx * self.context_len
        input_ids = torch.tensor(self.tokens[start_idx : start_idx + self.context_len])
        target_ids = torch.tensor(
            self.tokens[start_idx + 1 : start_idx + self.context_len + 1]
        )
        return input_ids, target_ids


def create_dataloaders(
    file_path, tokenizer, batch_size, context_len, train_ratio, token_mapping
):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    split_idx = int(len(text) * train_ratio)
    train_text = text[:split_idx]
    val_text = text[split_idx:]

    train_dataset = TextDataset(train_text, tokenizer, context_len, token_mapping)
    val_dataset = TextDataset(val_text, tokenizer, context_len, token_mapping)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, drop_last=False
    )

    return train_loader, val_loader


def reduce_vocab(file_path, tokenizer, top_k):
    print(f"Reading dataset from {file_path}...")
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    print("Tokenizing dataset...")
    tokens = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    print(f"Total tokens: {len(tokens):,}")

    print("Counting token frequencies...")
    token_counts = Counter(tokens)
    print(f"Unique tokens: {len(token_counts):,}")

    most_common = token_counts.most_common(top_k)
    top_tokens = [token for token, _ in most_common]
    print(f"Reduced vocabulary: {tokenizer.n_vocab:,} -> {len(top_tokens):,}")

    total_count = sum(count for _, count in most_common)
    coverage = 100 * total_count / len(tokens)
    print(f"Reduced coverage: {total_count:,} / {len(tokens):,} ({coverage:.1f}%)")

    token_mapping = {}
    for new_id, old_id in enumerate(top_tokens, start=1):  # 0 reserved for UNK token
        token_mapping[old_id] = new_id

    return token_mapping


def get_encoding(input_text, tokenizer, token_mapping):
    original_ids = tokenizer.encode(input_text, allowed_special={"<|endoftext|>"})
    return [token_mapping.get(token_id, 0) for token_id in original_ids]


def get_decoding(input_ids, tokenizer, inverse_token_mapping):
    original_ids = [
        inverse_token_mapping[token_id]
        for token_id in input_ids
        if token_id in inverse_token_mapping
    ]
    return tokenizer.decode(original_ids)


def get_parameter_counts(model):
    embedding_params = 0
    transformer_params = 0
    output_params = 0

    # embedding layers
    embedding_params += model.token_embedding.weight.numel()
    embedding_params += model.position_embedding.weight.numel()

    # transformer blocks
    for block in model.transformer_blocks:
        # multi-head attention
        transformer_params += block.attention.q_proj.weight.numel()
        transformer_params += block.attention.k_proj.weight.numel()
        transformer_params += block.attention.v_proj.weight.numel()
        transformer_params += block.attention.out_proj.weight.numel()
        transformer_params += block.attention.out_proj.bias.numel()

        # feed-forward network
        transformer_params += block.feed_forward.linear1.weight.numel()
        transformer_params += block.feed_forward.linear1.bias.numel()
        transformer_params += block.feed_forward.linear2.weight.numel()
        transformer_params += block.feed_forward.linear2.bias.numel()

        # layer normalisation
        transformer_params += block.ln1.weight.numel()
        transformer_params += block.ln1.bias.numel()
        transformer_params += block.ln2.weight.numel()
        transformer_params += block.ln2.bias.numel()

    # output layers
    output_params += model.ln_f.weight.numel()
    output_params += model.ln_f.bias.numel()
    output_params += model.lm_head.weight.numel()

    return embedding_params, transformer_params, output_params
