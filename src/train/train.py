import os
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from src.models.CNN import CNN


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    # Apple Silicon (MPS)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    project_root = Path(__file__).resolve().parents[2] 
    data_dir = project_root / "data" / "train"

    per_class = 1000
    train_ratio = 0.9
    epochs = 5
    batch_size = 128
    lr = 1e-3
    seed = 42

    random.seed(seed)
    torch.manual_seed(seed)

    device = get_device()
    print(f"Device: {device}")

    tfm = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
    ])

    # ImageFolder 会把 data/train/0..9 当成类别目录
    full_ds = datasets.ImageFolder(root=str(data_dir), transform=tfm)

    # -----------------------------
    # 每个类别随机抽 per_class 张，并按类别做 train/val 切分
    # -----------------------------
    class_indices = {c: [] for c in range(len(full_ds.classes))}
    for idx, (_, c) in enumerate(full_ds.samples):
        class_indices[c].append(idx)

    train_indices, val_indices = [], []
    for c, idxs in class_indices.items():
        if len(idxs) < per_class:
            raise ValueError(
                f"class '{full_ds.classes[c]}' only has {len(idxs)} images, < {per_class}"
            )
        random.shuffle(idxs)
        picked = idxs[:per_class]
        n_train = int(per_class * train_ratio)
        train_indices += picked[:n_train]
        val_indices += picked[n_train:]

    train_ds = Subset(full_ds, train_indices)
    val_ds = Subset(full_ds, val_indices)

    # macOS 上 num_workers>0 有时会有兼容问题；这里保守设 0（你需要更快可改成 2/4）
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False, num_workers=0)

    # -----------------------------
    # train
    # -----------------------------
    model = CNN(in_channels=1, num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

        # eval
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x).argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.numel()
        print(f"epoch {epoch+1}: val_acc = {correct/total:.4f}")

    # -----------------------------
    # save
    # -----------------------------
    ckpt_dir = project_root / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "cnn_custom.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"saved to {ckpt_path}")


if __name__ == "__main__":
    main()
"""Training script for the digit CNN model.

The script expects the training data to be arranged as:
data/train/
  0/*.png
  1/*.png
  ...
  9/*.png
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split

ROOT_DIR = Path(__file__).resolve().parents[1]  # points to repo/src
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from models.CNN import CNN  # noqa: E402


class DigitFolderDataset(Dataset):
    """Simple dataset reading digit images from class-named folders."""

    def __init__(self, root_dir: str, resize: int = 28):
        self.resize = resize
        self.samples: List[Tuple[str, int]] = []

        root_path = Path(root_dir)
        if not root_path.exists():
            raise FileNotFoundError(f"Data directory not found: {root_dir}")

        for label_dir in sorted(root_path.iterdir()):
            if not label_dir.is_dir() or not label_dir.name.isdigit():
                continue  # skip non-class folders such as .DS_Store
            label = int(label_dir.name)
            for img_path in label_dir.glob("*.*"):
                if img_path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                    continue
                self.samples.append((str(img_path), label))

        if not self.samples:
            raise RuntimeError(f"No images were found in {root_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"Failed to read image: {img_path}")

        if self.resize:
            img = cv2.resize(img, (self.resize, self.resize))

        img = torch.from_numpy(img).float() / 255.0  # scale to [0,1]
        img = img.unsqueeze(0)  # (1, H, W) for CNN in_channels=1
        return img, label


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), torch.tensor(targets, device=device, dtype=torch.long)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), torch.tensor(targets, device=device, dtype=torch.long)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    loss = running_loss / total
    acc = correct / total
    return loss, acc


def parse_args():
    parser = argparse.ArgumentParser(description="Train CNN on digit folder dataset.")
    parser.add_argument("--data-dir", type=str, default=str(Path("data") / "train"),
                        help="Root directory containing digit folders 0-9.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-split", type=float, default=0.1,
                        help="Fraction of data used for validation.")
    parser.add_argument("--output", type=str, default="checkpoints/best_model.pt",
                        help="Where to save the best model weights.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = DigitFolderDataset(args.data_dir)

    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = CNN(num_classes=10, in_channels=1)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_acc = 0.0
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch:02d}: "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), output_path)
            print(f"Saved new best model to {output_path}")

    print(f"Training finished. Best val acc: {best_val_acc:.3f}")


if __name__ == "__main__":
    main()
