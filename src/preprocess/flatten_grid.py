import argparse
from pathlib import Path

from PIL import Image

import cv2
import numpy as np


def _order_points(pts: np.ndarray) -> np.ndarray:
    pts = pts.astype("float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype="float32")


def _find_page_quad(bgr: np.ndarray) -> np.ndarray | None:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.dilate(edges, None, iterations=2)
    edges = cv2.erode(edges, None, iterations=1)

    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    for c in cnts[:10]:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            pts = approx.reshape(4, 2)
            return _order_points(pts)

    return None


def flatten_image_pil(
    img: Image.Image,
    cols: int = 9,
    rows: int = 13,
    cell_size: int = 64,
) -> Image.Image:
    bgr = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
    quad = _find_page_quad(bgr)

    h, w = bgr.shape[:2]
    if quad is None:
        quad = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype="float32")

    out_w = int(cols * cell_size)
    out_h = int(rows * cell_size)

    dst = np.array(
        [[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]],
        dtype="float32",
    )
    M = cv2.getPerspectiveTransform(quad, dst)
    warped = cv2.warpPerspective(bgr, M, (out_w, out_h), flags=cv2.INTER_CUBIC)

    rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def flatten_image_bytes(image_bytes: bytes, cols: int = 9, rows: int = 13, cell_size: int = 64) -> Image.Image:
    img = Image.open(Path(io.BytesIO(image_bytes))).convert("RGB")  # type: ignore
    return flatten_image_pil(img, cols=cols, rows=rows, cell_size=cell_size)


def flatten_image_path(
    image_path: Path,
    out_path: Path | None = None,
    cols: int = 9,
    rows: int = 13,
    cell_size: int = 64,
) -> Path:
    img = Image.open(image_path).convert("RGB")
    flat = flatten_image_pil(img, cols=cols, rows=rows, cell_size=cell_size)

    if out_path is None:
        out_path = image_path.with_name(image_path.stem + "_flat.png")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    flat.save(out_path)
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="path to photo (png/jpg)")
    parser.add_argument("--out", default=None, help="output path (default: <image>_flat.png)")
    parser.add_argument("--cols", type=int, default=9)
    parser.add_argument("--rows", type=int, default=13)
    parser.add_argument("--cell", type=int, default=64)
    args = parser.parse_args()

    in_path = Path(args.image)
    out_path = Path(args.out) if args.out else None
    saved = flatten_image_path(in_path, out_path=out_path, cols=args.cols, rows=args.rows, cell_size=args.cell)
    print(saved)


if __name__ == "__main__":
    main()