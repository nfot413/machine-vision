import argparse
import random
from pathlib import Path

import torch
from PIL import Image
from torchvision import datasets
from torch.utils.data import DataLoader

from src.utils.utils import device_auto, get_image_transform
from src.infer.infer import load_model  


@torch.no_grad()
def evaluate_testset(
    project_root: Path,
    ckpt_path: Path,
    batch_size: int = 256,
    seed: int = 42,
    device: torch.device | None = None,
    sample_per_class: int = 10,
    sample_seed: int | None = None,  # 样例采样的随机种子：None 表示每次都随机
):
    """
    返回：
    overall_acc: float
    per_class_acc: dict[int, float]
    samples: dict[int, list[tuple[Path, int]]]  # (图片路径, 预测标签)
    """
    device = device or device_auto()
    random.seed(seed)
    torch.manual_seed(seed)

    test_dir = project_root / "data" / "test"
    if not test_dir.exists():
        raise FileNotFoundError(f"test dir not found: {test_dir}")

    tfm = get_image_transform()
    ds = datasets.ImageFolder(root=str(test_dir), transform=tfm)  

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model, dev = load_model(ckpt_path, device=device)

    num_classes = len(ds.classes)
    correct = 0
    total = 0
    per_correct = [0] * num_classes
    per_total = [0] * num_classes

    for x, y in loader:
        x, y = x.to(dev), y.to(dev)
        pred = model(x).argmax(dim=1)
        match = (pred == y)

        correct += match.sum().item()
        total += y.numel()

        # per-class 统计
        for c in range(num_classes):
            mask = (y == c)
            per_total[c] += mask.sum().item()
            if mask.any():
                per_correct[c] += (match & mask).sum().item()

    overall_acc = correct / total if total > 0 else 0.0
    per_class_acc = {
        int(ds.classes[c]): (per_correct[c] / per_total[c] if per_total[c] > 0 else 0.0)
        for c in range(num_classes)
    }

    # ds.samples: list[(path, class_idx)]
    idx_by_class = {c: [] for c in range(num_classes)}
    for i, (p, c) in enumerate(ds.samples):
        idx_by_class[c].append(i)

    rng = random.Random(sample_seed) if sample_seed is not None else random.Random()  # 样例使用独立随机源

    samples = {int(ds.classes[c]): [] for c in range(num_classes)}
    for c in range(num_classes):
        idxs = idx_by_class[c][:]
        rng.shuffle(idxs)
        pick = idxs[:sample_per_class]

        for i in pick:
            img_path = Path(ds.samples[i][0])
            img = Image.open(img_path).convert("RGB")
            x = tfm(img).unsqueeze(0).to(dev)
            pred = int(model(x).argmax(dim=1).item())
            samples[int(ds.classes[c])].append((img_path, pred))

    return overall_acc, per_class_acc, samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=None, help="path to .pt (default: project_root/checkpoints/cnn.pt)")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    ckpt_path = Path(args.ckpt) if args.ckpt else (project_root / "checkpoints" / "cnn.pt")

    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    dev = device_auto()
    print("device:", dev)

    overall_acc, per_class_acc, _ = evaluate_testset(
        project_root=project_root,
        ckpt_path=ckpt_path,
        batch_size=args.batch_size,
        seed=args.seed,
        device=dev,
        sample_per_class=10,
    )

    print(f"\nOverall accuracy (5000 images): {overall_acc:.4f}")
    for k in range(10):
        print(f"Class {k} accuracy: {per_class_acc.get(k, 0.0):.4f}")


if __name__ == "__main__":
    main()