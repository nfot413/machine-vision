import argparse
from pathlib import Path
import io

from PIL import Image

import cv2
import numpy as np


def _order_points(pts: np.ndarray) -> np.ndarray:
    """排序四个顶点：左上、右上、右下、左下"""
    pts = pts.astype("float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]

    return np.array([tl, tr, br, bl], dtype="float32")


def _four_point_transform(image_bgr: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """透视变换：拉平图像"""
    rect = _order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = float(np.hypot(br[0] - bl[0], br[1] - bl[1]))
    widthB = float(np.hypot(tr[0] - tl[0], tr[1] - tl[1]))
    maxWidth = max(1, int(round(max(widthA, widthB))))

    heightA = float(np.hypot(tr[0] - br[0], tr[1] - br[1]))
    heightB = float(np.hypot(tl[0] - bl[0], tl[1] - bl[1]))
    maxHeight = max(1, int(round(max(heightA, heightB))))

    dst = np.array(
        [
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1],
        ],
        dtype="float32",
    )

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image_bgr, M, (maxWidth, maxHeight))
    return warped


def _extract_grid_bgr_from_bgr(img_bgr: np.ndarray) -> tuple[np.ndarray, bool]:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    thresh = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11,
        2,
    )

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img_bgr, False

    max_contour = max(contours, key=cv2.contourArea)

    epsilon = 0.02 * cv2.arcLength(max_contour, True)
    approx = cv2.approxPolyDP(max_contour, epsilon, True)

    if len(approx) == 4:
        pts = approx.reshape(4, 2).astype("float32")
        warped = _four_point_transform(img_bgr, pts)
        return warped, True

    x, y, w, h = cv2.boundingRect(max_contour)
    cropped = img_bgr[y : y + h, x : x + w]
    return cropped, False


def flatten_image_pil(img: Image.Image, cols: int = 9, rows: int = 13, cell_size: int = 64) -> Image.Image:
    bgr = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
    out_bgr, quad_found = _extract_grid_bgr_from_bgr(bgr)
    out_rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)

    out = Image.fromarray(out_rgb)
    out.info["quad_found"] = bool(quad_found)
    out.info["cols"] = int(cols)
    out.info["rows"] = int(rows)
    return out


def flatten_image_bytes(img_bytes: bytes, cols: int = 9, rows: int = 13, cell_size: int = 64) -> Image.Image:
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return flatten_image_pil(img, cols=cols, rows=rows, cell_size=cell_size)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="input photo path (png/jpg)")
    parser.add_argument("--out", default=None, help="output path (default: <image>_flat.png)")
    parser.add_argument("--cols", type=int, default=9)
    parser.add_argument("--rows", type=int, default=13)
    parser.add_argument("--cell_size", type=int, default=64)
    args = parser.parse_args()

    in_path = Path(args.image)
    out_path = Path(args.out) if args.out else in_path.with_name(in_path.stem + "_flat.png")

    img = Image.open(in_path).convert("RGB")
    flat = flatten_image_pil(img, cols=args.cols, rows=args.rows, cell_size=args.cell_size)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    flat.save(out_path)
    print(out_path)


if __name__ == "__main__":
    main()