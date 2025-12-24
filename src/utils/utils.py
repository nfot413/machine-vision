import torch
from torchvision import transforms

def device_auto() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")

def get_image_transform():
    return transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
    ])