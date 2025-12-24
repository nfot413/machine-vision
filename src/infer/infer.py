import io
from pathlib import Path
import argparse

import torch
import torch.nn.functional as F
from PIL import Image

from src.models.CNN import CNN
from src.utils.utils import device_auto, get_image_transform

_TFM = get_image_transform()


def load_model(ckpt_path: Path, device: torch.device | None = None) -> tuple[CNN, torch.device]:
    dev = device or device_auto()
    model = CNN().to(dev)
    model.load_state_dict(torch.load(ckpt_path, map_location=dev))
    model.eval()
    return model, dev


@torch.no_grad()
def predict_image_bytes(image_bytes: bytes, model: CNN, device: torch.device) -> int:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    x = _TFM(img).unsqueeze(0).to(device)
    logits = model(x)
    prob = F.softmax(logits, dim=1)[0]
    return int(prob.argmax().item())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="path to a png/jpg")
    parser.add_argument("--ckpt", default=None, help="path to .pt (default: project_root/checkpoints/cnn.pt)")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    ckpt_path = Path(args.ckpt) if args.ckpt else (root / "checkpoints" / "cnn.pt")

    dev = device_auto()
    print("device:", dev)

    model = CNN().to(dev)
    model.load_state_dict(torch.load(ckpt_path, map_location=dev))
    model.eval()

    img = Image.open(args.image).convert("RGB")
    x = _TFM(img).unsqueeze(0).to(dev)

    with torch.no_grad():
        logits = model(x)
        prob = F.softmax(logits, dim=1)[0]
        pred = int(prob.argmax().item())

    print("pred:", pred)


if __name__ == "__main__":
    main()