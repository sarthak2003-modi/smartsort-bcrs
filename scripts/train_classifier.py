#!/usr/bin/env python3
"""Train the transfer-learning classifier (freeze-then-finetune)."""

from __future__ import annotations

import argparse
from pathlib import Path

from smartsort import config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("data/processed"))
    p.add_argument("--backbone", default=config.DEFAULT_CLASSIFIER_BACKBONE)
    p.add_argument("--epochs-head", type=int, default=5)
    p.add_argument("--epochs-finetune", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr-head", type=float, default=1e-3)
    p.add_argument("--lr-finetune", type=float, default=1e-5)
    p.add_argument("--out", type=Path, default=Path("artifacts/classifier.pt"))
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader
        from torchvision import datasets, transforms
    except ImportError as exc:
        raise SystemExit(f"torch/torchvision required: {exc}") from exc

    from smartsort.models.classifier import TransferClassifier

    train_dir = args.data / "train"
    val_dir = args.data / "val"
    if not train_dir.is_dir():
        raise SystemExit(
            f"No training data at {train_dir}. Run scripts/prepare_dataset.py first. "
            "This repo does not ship a dataset."
        )

    tfm = transforms.Compose(
        [
            transforms.Resize((config.CLASSIFIER_IMAGE_SIZE, config.CLASSIFIER_IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    train_ds = datasets.ImageFolder(train_dir, transform=tfm)
    val_ds = datasets.ImageFolder(val_dir, transform=tfm) if val_dir.is_dir() else None
    class_ids = train_ds.classes
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = (
        DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
        if val_ds
        else None
    )

    clf = TransferClassifier(
        backbone=args.backbone,
        class_ids=class_ids,
        freeze_backbone=True,
        device=args.device,
    ).build()
    # Replace ImageNet transform with our folder transform path via raw model
    model = clf.model
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr_head)
    loss_fn = nn.CrossEntropyLoss()

    def run_epoch(loader, train: bool) -> float:
        model.train(train)
        total, correct, loss_sum = 0, 0, 0.0
        for x, y in loader:
            x, y = x.to(args.device), y.to(args.device)
            if train:
                opt.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            if train:
                loss.backward()
                opt.step()
            loss_sum += float(loss.item()) * len(y)
            correct += int((logits.argmax(1) == y).sum().item())
            total += len(y)
        acc = correct / max(1, total)
        print(f"{'train' if train else 'val'} loss={loss_sum/max(1,total):.4f} acc={acc:.4f}")
        return acc

    print(f"Stage 1: freeze backbone, train head ({args.epochs_head} epochs)")
    for epoch in range(args.epochs_head):
        print(f"epoch {epoch+1}/{args.epochs_head}")
        run_epoch(train_loader, True)
        if val_loader:
            run_epoch(val_loader, False)

    print(f"Stage 2: unfreeze, finetune ({args.epochs_finetune} epochs)")
    clf.unfreeze_backbone()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr_finetune)
    for epoch in range(args.epochs_finetune):
        print(f"epoch {epoch+1}/{args.epochs_finetune}")
        run_epoch(train_loader, True)
        if val_loader:
            run_epoch(val_loader, False)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    clf.save(args.out)
    meta = args.out.with_suffix(".classes.txt")
    meta.write_text("\n".join(class_ids) + "\n", encoding="utf-8")
    print(f"Saved {args.out} and {meta}")


if __name__ == "__main__":
    main()
