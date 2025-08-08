import torch
import time


def train_epoch(
    model,
    train_loader,
    optimizer,
    criterion,
    epoch,
    num_epochs,
    accumulation_steps,
    save_interval,
    model_path,
    device,
):
    model.train()
    total_loss = 0  # epoch loss
    start_time = time.time()

    for batch_idx, (input_ids, target_ids) in enumerate(train_loader):
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)

        logits = model(input_ids)

        # logits: (batch_size, seq_len, vocab_size) -> (batch_size * seq_len, vocab_size)
        # target_ids: (batch_size, seq_len) -> (batch_size * seq_len) to match CrossEntropyLoss expected input
        loss = criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
        total_loss += loss.item()

        print(
            f"Epoch {epoch + 1}/{num_epochs}, Batch {batch_idx + 1}/{len(train_loader)}, Train Loss: {loss.item():.4f}"
        )

        loss = loss / accumulation_steps
        loss.backward()

        if (batch_idx + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        # estimate time remaining for epoch
        elapsed_time = time.time() - start_time
        if batch_idx > 0:
            average_time = elapsed_time / (batch_idx + 1)
            remaining_batches = len(train_loader) - (batch_idx + 1)
            total_seconds = average_time * remaining_batches

            eta_h = int(total_seconds // 3600)
            eta_m = int((total_seconds % 3600) // 60)
            eta_s = int(total_seconds % 60)

            print(
                f"Estimated time remaining for epoch: {eta_h:02d}h {eta_m:02d}m {eta_s:02d}s"
            )

        # save model at specified batch intervals
        if (batch_idx + 1) % save_interval == 0:
            print(f"Saving model to {model_path}...")
            torch.save(model.state_dict(), model_path)
            print("Model saved successfully!")

    return total_loss / len(train_loader)


def validate(model, val_loader, criterion, device):
    print("Starting validation...")
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch_idx, (input_ids, target_ids) in enumerate(val_loader):
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)

            logits = model(input_ids)
            loss = criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
            total_loss += loss.item()

            print(
                f"Batch {batch_idx + 1}/{len(val_loader)}, Val Loss: {loss.item():.4f}"
            )

    return total_loss / len(val_loader)
