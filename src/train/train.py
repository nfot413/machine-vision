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
