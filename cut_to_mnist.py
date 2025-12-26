import cv2
import numpy as np
from PIL import Image
import os

# ---------------------- 配置参数（可直接修改） ----------------------
save_dir = "9x13_grid_mnist"  # 保存MNIST格式图片的文件夹
img_path = "clean_text.jpg"  # 本地图片路径
rows = 13  # 固定13行
cols = 9   # 固定9列
binary_threshold = 140  # 二值化阈值（白纸黑字）
canvas_size = 28  # MNIST画布尺寸
padding = 1       # 四周留白边距（像素），建议1~2，字符越大越好看
target_size = canvas_size - 2 * padding  # 目标最大字符尺寸，例如26

# ---------------------- 1. 初始化与图片读取 ----------------------
os.makedirs(save_dir, exist_ok=True)

img = cv2.imread(img_path)
if img is None:
    raise FileNotFoundError(f"未找到图片文件：{img_path}，请检查路径是否正确！")

# 灰度化 + 二值化（字符白、背景黑 → INV使黑字变白）
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, binary_threshold, 255, cv2.THRESH_BINARY_INV)

# ---------------------- 2. 按9列13行固定网格均分图片 ----------------------
img_h, img_w = thresh.shape
cell_w = img_w // cols
cell_h = img_h // rows

valid_count = 0

for row_idx in range(rows):
    y_start = row_idx * cell_h
    y_end = (row_idx + 1) * cell_h if row_idx < rows - 1 else img_h

    for col_idx in range(cols):
        x_start = col_idx * cell_w
        x_end = (col_idx + 1) * cell_w if col_idx < cols - 1 else img_w

        cell_roi = thresh[y_start:y_end, x_start:x_end]

        # 过滤全黑的空单元格
        if np.sum(cell_roi) == 0:
            continue

        # ---------------------- 3. 字符最大化放大 + 严格居中 ----------------------
        mnist_canvas = np.zeros((canvas_size, canvas_size), dtype=np.uint8)

        # 找到字符的有效边界（去除周围空白，更准确缩放）
        coords = np.column_stack(np.where(cell_roi > 0))
        if coords.size == 0:
            continue
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        char_crop = cell_roi[y_min:y_max+1, x_min:x_max+1]

        char_h, char_w = char_crop.shape

        # 计算缩放比例：让较长边达到 target_size
        scale = target_size / max(char_h, char_w)
        new_h = int(char_h * scale)
        new_w = int(char_w * scale)

        # 缩放（使用INTER_AREA保持清晰）
        resized_char = cv2.resize(char_crop, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # 计算居中偏移（保证不越界）
        y_offset = (canvas_size - new_h) // 2
        x_offset = (canvas_size - new_w) // 2

        # 放置到画布中心
        mnist_canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_char

        # ---------------------- 4. 保存图片 ----------------------
        save_path = os.path.join(save_dir, f"row{row_idx + 1}_col{col_idx + 1}_mnist_{valid_count}.png")
        Image.fromarray(mnist_canvas).save(save_path)
        valid_count += 1

# ---------------------- 结果输出 ----------------------
print(f"处理完成！")
print(f"1. 已按 {rows} 行 {cols} 列均分图片")
print(f"2. 字符已最大化放大（长边接近{target_size}像素，四周留{padding}px边距）并严格居中")
print(f"3. 共生成 {valid_count} 个MNIST格式字符图片，保存至 {save_dir} 文件夹")