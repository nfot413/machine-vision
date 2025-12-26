import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def _smooth_1d(x: np.ndarray, k: int) -> np.ndarray:
    k = max(1, int(k))
    if k == 1:
        return x
    kernel = np.ones(k, dtype=np.float32) / k
    return np.convolve(x.astype(np.float32), kernel, mode="same")


def _pick_line_centers(proj: np.ndarray, n_lines: int) -> list[int]:
    proj = proj.astype(np.float32)
    proj = (proj - proj.min()) / (proj.max() - proj.min() + 1e-6)
    proj_s = _smooth_1d(proj, max(3, len(proj) // (n_lines * 8)))

    thr = float(proj_s.mean() + 0.6 * proj_s.std())
    idx = np.where(proj_s > thr)[0]

    centers: list[int] = []
    if idx.size > 0:
        start = idx[0]
        prev = idx[0]
        for i in idx[1:]:
            if i == prev + 1:
                prev = i
            else:
                seg = np.arange(start, prev + 1)
                w = proj_s[seg]
                c = int(np.round((seg * w).sum() / (w.sum() + 1e-6)))
                centers.append(c)
                start = i
                prev = i
        seg = np.arange(start, prev + 1)
        w = proj_s[seg]
        c = int(np.round((seg * w).sum() / (w.sum() + 1e-6)))
        centers.append(c)

    L = len(proj)
    if len(centers) < max(2, n_lines // 2):
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


def _cell_to_mnist28_blackbg(cell_rgb: Image.Image) -> Image.Image:
    g = cell_rgb.convert("L")
    a = np.array(g).astype(np.uint8)

    b = max(1, min(a.shape[0], a.shape[1]) // 25)
    a[:b, :] = 255
    a[-b:, :] = 255
    a[:, :b] = 255
    a[:, -b:] = 255

    thr = float(np.clip(a.mean() - 0.5 * a.std(), 40, 220))
    mask = a < thr

    if mask.sum() < 15:
        return Image.fromarray(np.zeros((28, 28), dtype=np.uint8), mode="L")

    ys, xs = np.where(mask)
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1

    crop_mask = mask[y0:y1, x0:x1].astype(np.uint8) * 255  # 白字
    h, w = crop_mask.shape[:2]
    side = int(max(h, w))

    canvas = np.zeros((side, side), dtype=np.uint8)  # 黑底
    oy = (side - h) // 2
    ox = (side - w) // 2
    canvas[oy:oy + h, ox:ox + w] = crop_mask

    if side <= 0:
        return Image.fromarray(np.zeros((28, 28), dtype=np.uint8), mode="L")

    pil_sq = Image.fromarray(canvas, mode="L")
    pil_20 = pil_sq.resize((20, 20), Image.BILINEAR)

    out = Image.fromarray(np.zeros((28, 28), dtype=np.uint8), mode="L")
    out.paste(pil_20, (4, 4))
    return out


def split_flattened_pil(
    flat_img: Image.Image,
    cols: int = 9,
    rows: int = 13,
    inner_pad: float = 0.10,
) -> list[Image.Image]:
    img = flat_img.convert("RGB")
    w, h = img.size

    gray = np.array(img.convert("L")).astype(np.uint8)
    t_line = int(np.clip(np.percentile(gray, 25), 30, 120))
    line_mask = (gray < t_line).astype(np.uint8)

    vx = line_mask.sum(axis=0)
    hy = line_mask.sum(axis=1)

    v_lines = _pick_line_centers(vx, cols + 1)
    h_lines = _pick_line_centers(hy, rows + 1)

    digits: list[Image.Image] = []
    for r in range(rows):
        for c in range(cols):
            x0, x1 = v_lines[c], v_lines[c + 1]
            y0, y1 = h_lines[r], h_lines[r + 1]

            cw = max(1, x1 - x0)
            ch = max(1, y1 - y0)
            px = int(cw * inner_pad)
            py = int(ch * inner_pad)

            xx0 = int(np.clip(x0 + px, 0, w - 1))
            xx1 = int(np.clip(x1 - px, xx0 + 1, w))
            yy0 = int(np.clip(y0 + py, 0, h - 1))
            yy1 = int(np.clip(y1 - py, yy0 + 1, h))

            cell = img.crop((xx0, yy0, xx1, yy1))
            digits.append(_cell_to_mnist28_blackbg(cell))

    return digits


def split_flattened_path(
    image_path: Path,
    out_dir: Path,
    cols: int = 9,
    rows: int = 13,
    inner_pad: float = 0.10,
) -> Path:
    img = Image.open(image_path).convert("RGB")
    digits = split_flattened_pil(img, cols=cols, rows=rows, inner_pad=inner_pad)

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
    args = parser.parse_args()

    out = split_flattened_path(
        Path(args.image),
        Path(args.out_dir),
        cols=args.cols,
        rows=args.rows,
        inner_pad=args.inner_pad,
    )
    print(out)


if __name__ == "__main__":
    main()