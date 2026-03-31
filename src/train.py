"""
Training script for GPT2ForSequenceClassification on 20 Newsgroups dataset.

This script trains a GPT-2 based classifier without using any HuggingFace libraries.
"""

import argparse
import json
import os
import time
from typing import List, Dict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from gpt2 import GPT2ForSequenceClassification


class NewsGroupsDataset(Dataset):
    def __init__(self, jsonl_path: str, max_length: int = 256):
        self.examples: List[Dict] = []
        self.max_length = max_length

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                ex = json.loads(line)
                token_ids = ex["token_ids"][:self.max_length]
                label = ex["label"]

                self.examples.append({
                    "input_ids": token_ids,
                    "label": label,
                })

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ex = self.examples[idx]
        return {
            "input_ids": torch.tensor(ex["input_ids"], dtype=torch.long),
            "label": torch.tensor(ex["label"], dtype=torch.long),
        }


def collate_batch(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    input_ids_list = [item["input_ids"] for item in batch]
    labels = torch.stack([item["label"] for item in batch])

    max_len = max(x.size(0) for x in input_ids_list)

    padded_input_ids = []
    for x in input_ids_list:
        pad_len = max_len - x.size(0)
        if pad_len > 0:
            pad = torch.zeros(pad_len, dtype=torch.long)
            x = torch.cat([pad, x], dim=0)  # left pad
        padded_input_ids.append(x)

    input_ids = torch.stack(padded_input_ids)

    return {
        "input_ids": input_ids,
        "labels": labels,
    }


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float = 1.0,
) -> float:
    model.train()

    total_loss = 0.0
    total_examples = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids)
        logits = outputs.logits
        loss = criterion(logits, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_examples += batch_size

    return total_loss / total_examples


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()

    total_loss = 0.0
    total_examples = 0
    total_correct = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids)
        logits = outputs.logits
        loss = criterion(logits, labels)

        preds = torch.argmax(logits, dim=-1)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_examples += batch_size
        total_correct += (preds == labels).sum().item()

    avg_loss = total_loss / total_examples
    accuracy = total_correct / total_examples

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
    }


def save_metrics(metrics: Dict[str, List[float]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default="data/20_newsgroups_train.jsonl")
    parser.add_argument("--val_path", type=str, default="data/20_newsgroups_val.jsonl")
    parser.add_argument("--lm_checkpoint", type=str, default="checkpoints/gpt2_model.pth")
    parser.add_argument("--save_path", type=str, default="checkpoints/classifier_model.pth")
    parser.add_argument("--metrics_path", type=str, default="checkpoints/training_metrics.json")
    parser.add_argument("--log_dir", type=str, default="runs/gpt2_classifier")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    args = parser.parse_args()

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    train_dataset = NewsGroupsDataset(args.train_path, max_length=args.max_length)
    val_dataset = NewsGroupsDataset(args.val_path, max_length=args.max_length)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_batch,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_batch,
    )

    print(f"Loaded {len(train_dataset)} training examples")
    print(f"Loaded {len(val_dataset)} validation examples")

    model = GPT2ForSequenceClassification(
        lm_bin_path=args.lm_checkpoint
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    writer = SummaryWriter(log_dir=args.log_dir)

    metrics = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
    }

    best_val_acc = 0.0
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            grad_clip=args.grad_clip,
        )

        val_metrics = evaluate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
        )

        val_loss = val_metrics["loss"]
        val_acc = val_metrics["accuracy"]

        metrics["train_loss"].append(train_loss)
        metrics["val_loss"].append(val_loss)
        metrics["val_accuracy"].append(val_acc)

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Accuracy/val", val_acc, epoch)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), args.save_path)
            print(f"Saved new best model to {args.save_path}")

        epoch_time = time.time() - epoch_start
        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_acc:.4f} | "
            f"time={epoch_time:.1f}s"
        )

    total_time = time.time() - start_time
    save_metrics(metrics, args.metrics_path)
    writer.close()

    print(f"Training complete in {total_time:.1f}s")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Saved metrics to {args.metrics_path}")
    print(f"Best checkpoint saved to {args.save_path}")


with open("src/train.py", "w", encoding="utf-8") as f:
    f.write(train_code)

print("Updated src/train.py")