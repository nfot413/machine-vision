import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

from src.models.CNN import CNN
from src.utils.utils import device_auto, get_image_transform


@torch.no_grad()
def _eval_acc(model, loader, device) -> float:
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / total


def train_and_save(
    project_root: Path | None = None,
    per_class: int = 1000,
    train_ratio: float = 0.9,
    epochs: int = 5,
    batch_size: int = 128,
    lr: float = 1e-3,
    seed: int = 42,
    device: torch.device | None = None,
    progress_cb=None,
):
    project_root = project_root or Path(__file__).resolve().parents[2]
    data_dir = project_root / "data" / "train"

    random.seed(seed)
    torch.manual_seed(seed)

    device = device or device_auto()

    tfm = get_image_transform()
    full_ds = datasets.ImageFolder(root=str(data_dir), transform=tfm)

    class_indices = {c: [] for c in range(len(full_ds.classes))}
    for idx, (_, c) in enumerate(full_ds.samples):
        class_indices[c].append(idx)

    train_indices, val_indices = [], []
    for c, idxs in class_indices.items():
        if len(idxs) < per_class:
            raise ValueError(f"class '{full_ds.classes[c]}' only has {len(idxs)} images, < {per_class}")
        random.shuffle(idxs)
        picked = idxs[:per_class]
        n_train = int(per_class * train_ratio)
        train_indices += picked[:n_train]
        val_indices += picked[n_train:]

    train_ds = Subset(full_ds, train_indices)
    val_ds = Subset(full_ds, val_indices)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False, num_workers=0)

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

        acc = _eval_acc(model, val_loader, device)
        if progress_cb:
            progress_cb(epoch + 1, epochs, acc)
        else:
            print(f"epoch {epoch+1}: val_acc = {acc:.4f}")

    ckpt_dir = project_root / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "cnn.pt"
    torch.save(model.state_dict(), ckpt_path)

    return model, ckpt_path, device


def main():
    model, ckpt_path, device = train_and_save()
    print(f"Device: {device}")
    print(f"saved to {ckpt_path}")


if __name__ == "__main__":
    main()