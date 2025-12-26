import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def _smooth_1d(x: np.ndarray, k: int) -> np.ndarray:
    k = max(1, int(k))
    if k == 1:
        return x.astype(np.float32)
    kernel = np.ones(k, dtype=np.float32) / k
    return np.convolve(x.astype(np.float32), kernel, mode="same")


def _pick_lines_from_projection(proj: np.ndarray, n_lines: int) -> list[int]:
    L = len(proj)
    if n_lines <= 2 or L < 2:
        return [0, max(0, L - 1)]

    p = proj.astype(np.float32)
    if p.max() <= 0:
        step = (L - 1) / (n_lines - 1)
        return [int(round(i * step)) for i in range(n_lines)]

    p = (p - p.min()) / (p.max() - p.min() + 1e-6)
    p = _smooth_1d(p, max(5, L // (n_lines * 12)))

    thr = float(p.mean() + 0.8 * p.std())
    idx = np.where(p > thr)[0]

    centers: list[int] = []
    if idx.size > 0:
        start = int(idx[0])
        prev = int(idx[0])
        for i in idx[1:]:
            i = int(i)
            if i == prev + 1:
                prev = i
            else:
                seg = np.arange(start, prev + 1)
                w = p[seg]
                c = int(np.round((seg * w).sum() / (w.sum() + 1e-6)))
                centers.append(c)
                start = i
                prev = i
        seg = np.arange(start, prev + 1)
        w = p[seg]
        c = int(np.round((seg * w).sum() / (w.sum() + 1e-6)))
        centers.append(c)

    if len(centers) < n_lines:
        step = (L - 1) / (n_lines - 1)
        return [int(round(i * step)) for i in range(n_lines)]

    centers = sorted(set(int(c) for c in centers))
    step = (L - 1) / (n_lines - 1)

    picked: list[int] = []
    used = set()
    for i in range(n_lines):
        target = i * step
        best = None
        best_d = 1e18
        for c in centers:
            if c in used:
                continue
            d = abs(c - target)
            if d < best_d:
                best_d = d
                best = c
        if best is None:
            best = int(round(target))
        used.add(best)
        picked.append(int(best))

    picked = sorted(picked)
    picked[0] = 0
    picked[-1] = L - 1
    return picked


def _remove_border_components(bin_img: np.ndarray) -> np.ndarray:
    if bin_img is None or bin_img.size == 0:
        return bin_img

    x = (bin_img > 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(x, connectivity=8)
    if n <= 1:
        return bin_img

    H, W = x.shape[:2]
    keep = np.zeros_like(x)

    small_area_thr = max(60, int(0.01 * H * W))

    for cid in range(1, n):
        x0, y0, w, h, area = stats[cid]
        x1 = x0 + w - 1
        y1 = y0 + h - 1
        touches = (x0 <= 0) or (y0 <= 0) or (x1 >= W - 1) or (y1 >= H - 1)

        if touches and area <= small_area_thr:
            continue

        keep[labels == cid] = 255

    return keep


def _rect_union(b1: tuple[int, int, int, int], b2: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    x0 = min(b1[0], b2[0])
    y0 = min(b1[1], b2[1])
    x1 = max(b1[2], b2[2])
    y1 = max(b1[3], b2[3])
    return (x0, y0, x1, y1)


def _rect_linf_distance(b1: tuple[int, int, int, int], b2: tuple[int, int, int, int]) -> int:
    x0a, y0a, x1a, y1a = b1
    x0b, y0b, x1b, y1b = b2

    dx = 0
    if x1a < x0b:
        dx = x0b - x1a - 1
    elif x1b < x0a:
        dx = x0a - x1b - 1

    dy = 0
    if y1a < y0b:
        dy = y0b - y1a - 1
    elif y1b < y0a:
        dy = y0a - y1b - 1

    return int(max(dx, dy))


def _keep_largest_component(bin_img: np.ndarray, min_area: int = 40) -> np.ndarray:
    x = (bin_img > 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(x, connectivity=8)
    if n <= 1:
        return bin_img * 0

    # 找最大连通域
    best = -1
    best_area = 0
    for cid in range(1, n):
        area = int(stats[cid, cv2.CC_STAT_AREA])
        if area > best_area:
            best_area = area
            best = cid

    if best < 0 or best_area < min_area:
        return bin_img * 0

    H, W = x.shape[:2]

    def bbox(cid: int) -> tuple[int, int, int, int]:
        x0 = int(stats[cid, cv2.CC_STAT_LEFT])
        y0 = int(stats[cid, cv2.CC_STAT_TOP])
        w = int(stats[cid, cv2.CC_STAT_WIDTH])
        h = int(stats[cid, cv2.CC_STAT_HEIGHT])
        return (x0, y0, x0 + w - 1, y0 + h - 1)

    keep_ids = {best}
    keep_box = bbox(best)

    gap = max(3, int(0.08 * max(H, W)))

    area_thr = max(5, min_area // 4, int(0.01 * best_area))

    changed = True
    while changed:
        changed = False
        for cid in range(1, n):
            if cid in keep_ids:
                continue
            area = int(stats[cid, cv2.CC_STAT_AREA])
            if area < area_thr:
                continue

            cb = bbox(cid)
            dist = _rect_linf_distance(keep_box, cb)
            if dist <= gap:
                keep_ids.add(cid)
                keep_box = _rect_union(keep_box, cb)
                changed = True

    out = np.zeros_like(x, dtype=np.uint8)
    for cid in keep_ids:
        out[labels == cid] = 255
    return out


def _to_mnist28(bin_char: np.ndarray, canvas_size: int = 28, padding: int = 1) -> Image.Image:
    if bin_char is None or bin_char.size == 0:
        return Image.fromarray(np.zeros((canvas_size, canvas_size), dtype=np.uint8), mode="L")

    coords = np.column_stack(np.where(bin_char > 0))
    if coords.size == 0:
        return Image.fromarray(np.zeros((canvas_size, canvas_size), dtype=np.uint8), mode="L")

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    crop = bin_char[y_min:y_max + 1, x_min:x_max + 1]

    ch, cw = crop.shape[:2]
    if ch <= 0 or cw <= 0:
        return Image.fromarray(np.zeros((canvas_size, canvas_size), dtype=np.uint8), mode="L")

    target = canvas_size - 2 * padding
    scale = float(target) / float(max(ch, cw))
    new_h = max(1, int(round(ch * scale)))
    new_w = max(1, int(round(cw * scale)))

    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    canvas = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
    y_off = (canvas_size - new_h) // 2
    x_off = (canvas_size - new_w) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized

    return Image.fromarray(canvas, mode="L")


def split_flattened_pil(
    flat_img: Image.Image,
    cols: int = 9,
    rows: int = 13,
    inner_pad: float = 0.10,
    min_area: int = 40,
    padding: int = 1,
) -> list[Image.Image]:
    img = np.array(flat_img.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    bw = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 2
    )

    H, W = bw.shape[:2]
    cell_w = max(1, int(round(W / cols)))
    cell_h = max(1, int(round(H / rows)))

    kx = max(25, int(cell_w * 0.80))
    ky = max(25, int(cell_h * 0.80))
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kx, 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, ky))

    h_lines = cv2.morphologyEx(bw, cv2.MORPH_OPEN, h_kernel, iterations=1)
    v_lines = cv2.morphologyEx(bw, cv2.MORPH_OPEN, v_kernel, iterations=1)
    grid = cv2.bitwise_or(h_lines, v_lines)

    digits = cv2.subtract(bw, grid)

    vx = (grid > 0).sum(axis=0)
    hy = (grid > 0).sum(axis=1)

    v_lines_pos = _pick_lines_from_projection(vx, cols + 1)
    h_lines_pos = _pick_lines_from_projection(hy, rows + 1)

    out: list[Image.Image] = []
    for r in range(rows):
        for c in range(cols):
            x0, x1 = v_lines_pos[c], v_lines_pos[c + 1]
            y0, y1 = h_lines_pos[r], h_lines_pos[r + 1]

            cw = max(1, x1 - x0)
            ch = max(1, y1 - y0)
            px = int(cw * inner_pad)
            py = int(ch * inner_pad)

            xx0 = int(np.clip(x0 + px, 0, W - 1))
            xx1 = int(np.clip(x1 - px, xx0 + 1, W))
            yy0 = int(np.clip(y0 + py, 0, H - 1))
            yy1 = int(np.clip(y1 - py, yy0 + 1, H))

            roi = digits[yy0:yy1, xx0:xx1].copy()

            if roi.shape[0] > 2 and roi.shape[1] > 2:
                roi[0:1, :] = 0
                roi[-1:, :] = 0
                roi[:, 0:1] = 0
                roi[:, -1:] = 0

            roi = _remove_border_components(roi)
            roi = _keep_largest_component(roi, min_area=min_area)

            if roi.sum() == 0:
                out.append(Image.fromarray(np.zeros((28, 28), dtype=np.uint8), mode="L"))
                continue

            roi = cv2.morphologyEx(
                roi,
                cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)),
                iterations=1
            )

            out.append(_to_mnist28(roi, canvas_size=28, padding=padding))

    return out


def split_flattened_path(
    image_path: Path,
    out_dir: Path,
    cols: int = 9,
    rows: int = 13,
    inner_pad: float = 0.10,
    min_area: int = 40,
    padding: int = 1,
) -> Path:
    img = Image.open(image_path).convert("RGB")
    digits = split_flattened_pil(
        img,
        cols=cols,
        rows=rows,
        inner_pad=inner_pad,
        min_area=min_area,
        padding=padding,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    for i, d in enumerate(digits):
        r = i // cols
        c = i % cols
        d.save(out_dir / f"r{r:02d}_c{c:02d}.png")

    return out_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="path to flattened image (png/jpg)")
    parser.add_argument("--out_dir", required=True, help="output directory for 28x28 images")
    parser.add_argument("--cols", type=int, default=9)
    parser.add_argument("--rows", type=int, default=13)
    parser.add_argument("--inner_pad", type=float, default=0.10)
    parser.add_argument("--min_area", type=int, default=40)
    parser.add_argument("--padding", type=int, default=1)
    args = parser.parse_args()

    out = split_flattened_path(
        Path(args.image),
        Path(args.out_dir),
        cols=args.cols,
        rows=args.rows,
        inner_pad=args.inner_pad,
        min_area=args.min_area,
        padding=args.padding,
    )
    print(out)


if __name__ == "__main__":
    main()