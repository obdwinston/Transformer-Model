import os
import torch
import torch.nn as nn
import tiktoken
import json
from model import GPT2
from train import train_epoch, validate
from utils import (
    create_dataloaders,
    reduce_vocab,
    get_encoding,
    get_decoding,
    get_parameter_counts,
)

# model constants
VOCAB_SIZE = 8000
CONTEXT_LEN = 512
WINDOW_SIZE = 256
EMBED_DIM = 768
NUM_HEADS = 16
NUM_LAYERS = 4
FF_DIM = 4 * EMBED_DIM
DROPOUT = 0.0

# training constants
BATCH_SIZE = 16
ACCUMULATION_STEPS = 5
TRAIN_RATIO = 0.9
LEARNING_RATE = 5e-4
BETA_1 = 0.9
BETA_2 = 0.95
WEIGHT_DECAY = 0.1
NUM_EPOCHS = 1
DATA_PATH = "data.txt"
MODEL_PATH = "model.pth"
TOKEN_MAPPING_PATH = "token_mapping.json"
SAVE_INTERVAL = 1000 * ACCUMULATION_STEPS
DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# generation constants
MAX_NEW_TOKENS = 100
TEMPERATURE = 0.2
TOP_K = None
TOP_P = 0.9


def main():
    print(f"Using device: {DEVICE}")

    tokenizer = tiktoken.get_encoding("gpt2")

    if os.path.exists(TOKEN_MAPPING_PATH):
        print(f"Loading token mapping from {TOKEN_MAPPING_PATH}...")
        with open(TOKEN_MAPPING_PATH, "r") as f:
            token_mapping = {int(k): v for k, v in json.load(f).items()}
        print("Token mapping loaded successfully!")
    else:
        print("Creating reduced vocabulary...")
        token_mapping = reduce_vocab(DATA_PATH, tokenizer, top_k=VOCAB_SIZE)
        print(f"Saving token mapping to {TOKEN_MAPPING_PATH}...")
        with open(TOKEN_MAPPING_PATH, "w") as f:
            json.dump(token_mapping, f)
        print("Token mapping saved successfully!")

    reduced_vocab_size = len(token_mapping) + 1  # +1 for reserved UNK token
    inverse_token_mapping = {v: k for k, v in token_mapping.items()}  # new to old

    model = GPT2(
        vocab_size=reduced_vocab_size,
        context_len=CONTEXT_LEN,
        window_size=WINDOW_SIZE,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        ff_dim=FF_DIM,
        dropout=DROPOUT,
    ).to(DEVICE)

    embedding_params, transformer_params, output_params = get_parameter_counts(model)
    total_params = embedding_params + transformer_params + output_params

    print("Model parameters:")
    print(
        f"  Embedding layers: {embedding_params:,} ({embedding_params / total_params * 100:.1f}%)"
    )
    print(
        f"  Transformer blocks: {transformer_params:,} ({transformer_params / total_params * 100:.1f}%)"
    )
    print(
        f"  Output layers: {output_params:,} ({output_params / total_params * 100:.1f}%)"
    )
    print(f"  Total parameters: {total_params:,}")
    print(f"Model architecture:\n{model}\n")

    # TRAINING

    if os.path.exists(MODEL_PATH):
        print(f"Loading model from {MODEL_PATH}...")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print("Model loaded successfully!")
    else:
        print("Starting training...")

        train_loader, val_loader = create_dataloaders(
            DATA_PATH, tokenizer, BATCH_SIZE, CONTEXT_LEN, TRAIN_RATIO, token_mapping
        )

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
            betas=(BETA_1, BETA_2),
            weight_decay=WEIGHT_DECAY,
        )
        criterion = nn.CrossEntropyLoss()

        print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

        for epoch in range(NUM_EPOCHS):
            train_loss = train_epoch(
                model=model,
                train_loader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                epoch=epoch,
                num_epochs=NUM_EPOCHS,
                accumulation_steps=ACCUMULATION_STEPS,
                save_interval=SAVE_INTERVAL,
                model_path=MODEL_PATH,
                device=DEVICE,
            )
            val_loss = validate(model, val_loader, criterion, DEVICE)
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        print(f"Saving model to {MODEL_PATH}...")
        torch.save(model.state_dict(), MODEL_PATH)
        print("Model saved successfully!")

    # GENERATION

    model.eval()
    prompt = "Once upon a time in a land far, far away, there lived a"

    input_ids = get_encoding(prompt, tokenizer, token_mapping)
    input_tensor = torch.tensor([input_ids]).to(DEVICE)

    response = model.generate(
        input_tensor,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_k=TOP_K,
        top_p=TOP_P,
    )

    output_ids = response[0].tolist()
    output_text = get_decoding(output_ids, tokenizer, inverse_token_mapping)
    print(f"Output: {output_text}")


if __name__ == "__main__":
    main()
