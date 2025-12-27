import argparse
from pathlib import Path

from PIL import Image

import cv2
import numpy as np


def remove_grid_lines(image_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)

    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=2)

    grid_lines = cv2.add(horizontal_lines, vertical_lines)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    grid_lines_dilated = cv2.dilate(grid_lines, kernel, iterations=1)

    binary_clean = binary.copy()
    binary_clean[grid_lines_dilated > 0] = 0
    return binary_clean


def _slice_grid_to_pil_list(
    binary_img: np.ndarray,
    rows: int = 13,
    cols: int = 9,
    canvas_size: int = 28,
    padding: int = 4,
    min_area: int = 15,
) -> list[Image.Image]:
    img_h, img_w = binary_img.shape
    cell_w = max(1, img_w // cols)
    cell_h = max(1, img_h // rows)
    target_size = canvas_size - 2 * padding

    digits_list: list[Image.Image] = []
    blank_canvas = Image.new("L", (canvas_size, canvas_size), 0)

    for row_idx in range(rows):
        y_start = row_idx * cell_h
        y_end = (row_idx + 1) * cell_h if row_idx < rows - 1 else img_h

        for col_idx in range(cols):
            x_start = col_idx * cell_w
            x_end = (col_idx + 1) * cell_w if col_idx < cols - 1 else img_w

            out_img = blank_canvas

            margin = 4
            if (y_end - y_start) > 2 * margin and (x_end - x_start) > 2 * margin:
                cell_roi = binary_img[y_start + margin : y_end - margin, x_start + margin : x_end - margin]
            else:
                cell_roi = binary_img[y_start:y_end, x_start:x_end]

            if cv2.countNonZero(cell_roi) < 10:
                digits_list.append(out_img)
                continue

            cnts, _ = cv2.findContours(cell_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                digits_list.append(out_img)
                continue

            valid_cnts = [c for c in cnts if cv2.contourArea(c) > float(max(1, min_area))]
            if not valid_cnts:
                digits_list.append(out_img)
                continue

            all_points = np.vstack(valid_cnts)
            x, y, w, h = cv2.boundingRect(all_points)
            char_crop = cell_roi[y : y + h, x : x + w]

            if char_crop.size == 0:
                digits_list.append(out_img)
                continue

            ch, cw = char_crop.shape
            if ch <= 0 or cw <= 0:
                digits_list.append(out_img)
                continue

            scale = float(target_size) / float(max(ch, cw))
            new_h = max(1, int(round(ch * scale)))
            new_w = max(1, int(round(cw * scale)))

            resized_char = cv2.resize(char_crop, (new_w, new_h), interpolation=cv2.INTER_AREA)

            mnist_canvas = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
            y_offset = (canvas_size - new_h) // 2
            x_offset = (canvas_size - new_w) // 2
            mnist_canvas[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized_char

            out_img = Image.fromarray(mnist_canvas, mode="L")
            digits_list.append(out_img)

    return digits_list


def split_flattened_pil(
    flat_img: Image.Image,
    cols: int = 9,
    rows: int = 13,
    inner_pad: float = 0.10,
    min_area: int = 15,
    padding: int = 4,
) -> list[Image.Image]:

    bgr = cv2.cvtColor(np.array(flat_img.convert("RGB")), cv2.COLOR_RGB2BGR)
    binary_clean = remove_grid_lines(bgr)

    digits = _slice_grid_to_pil_list(
        binary_clean,
        rows=int(rows),
        cols=int(cols),
        canvas_size=28,
        padding=int(padding),
        min_area=int(min_area),
    )
    return digits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="path to flattened image (png/jpg)")
    parser.add_argument("--out_dir", required=True, help="output directory for 117 MNIST images")
    parser.add_argument("--cols", type=int, default=9)
    parser.add_argument("--rows", type=int, default=13)
    parser.add_argument("--inner_pad", type=float, default=0.10)
    parser.add_argument("--min_area", type=int, default=15)
    parser.add_argument("--padding", type=int, default=4)
    args = parser.parse_args()

    in_path = Path(args.image)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    img = Image.open(in_path).convert("RGB")
    digits = split_flattened_pil(
        img,
        cols=args.cols,
        rows=args.rows,
        inner_pad=args.inner_pad,
        min_area=args.min_area,
        padding=args.padding,
    )

    for i, d in enumerate(digits):
        r = i // args.cols
        c = i % args.cols
        d.save(out_dir / f"r{r:02d}_c{c:02d}.png")

    print(out_dir)


if __name__ == "__main__":
    main()