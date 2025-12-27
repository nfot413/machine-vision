import cv2
import numpy as np
from PIL import Image
import os
import random
import glob  # 新增：用于文件模式匹配

# --- 配置区域 ---
# 定义要寻找的文件模式，例如 "4_*.jpg" 会匹配 4_1.jpg, 4_2.jpg, 4_test.jpg 等
input_pattern = "4_*.jpg"
save_dir = "four_mnist_x4_combo"  # 保存目录
os.makedirs(save_dir, exist_ok=True)


# ----------------

# ==========================================
# --- 核心工具函数 (与之前保持一致) ---
# ==========================================

def resize_and_pad(roi, target_size):
    """缩放并居中到28x28画布"""
    h, w = roi.shape
    canvas = np.zeros((28, 28), dtype=np.uint8)
    if w == 0 or h == 0: return canvas
    scale = min(target_size / w, target_size / h)
    new_w, new_h = int(w * scale), int(h * scale)
    new_w, new_h = max(1, new_w), max(1, new_h)
    # 使用 INTER_AREA 插值，对于数字4这种有交叉点的结构效果较好
    resized = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_AREA)
    x_pad = (28 - new_w) // 2
    y_pad = (28 - new_h) // 2
    h_end = min(y_pad + new_h, 28)
    w_end = min(x_pad + new_w, 28)
    canvas[y_pad:h_end, x_pad:w_end] = resized[:h_end - y_pad, :w_end - x_pad]
    return canvas


def apply_rotation(image, angle):
    """旋转图片 (背景保持黑色)"""
    h, w = image.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return rotated


def apply_thickness(image):
    """随机变粗、变细或保持不变"""
    # 数字4的笔画交叉处容易在变细时断裂，所以增加保持原样的概率
    mode = random.choice(['thicker', 'thinner', 'none', 'none'])
    kernel = np.ones((2, 2), np.uint8)
    if mode == 'thicker':
        return cv2.dilate(image, kernel, iterations=1)
    elif mode == 'thinner':
        return cv2.erode(image, kernel, iterations=1)
    else:
        return image


# ==========================================
# --- 主处理流程 (批量文件循环) ---
# ==========================================

# 1. 查找所有符合模式的输入文件
input_files = glob.glob(input_pattern)
input_files.sort()  # 排序，保证处理顺序

if not input_files:
    print(f"错误：在当前目录下未找到符合 '{input_pattern}' 的文件。")
    exit()

print(f"找到 {len(input_files)} 个文件待处理: {input_files}")

total_digits_found = 0
total_images_generated = 0

# --- 外层循环：遍历每个文件 (如 4_1.jpg, 4_2.jpg) ---
for filepath in input_files:
    # 获取不带扩展名的文件名 (例如 "4_1")，用于生成输出文件名
    base_name = os.path.splitext(os.path.basename(filepath))[0]
    print(f"\n--- 正在处理文件: {filepath} ---")

    img = cv2.imread(filepath)
    if img is None:
        print(f"警告：无法读取 {filepath}，跳过。")
        continue

    # 预处理
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 阈值：数字4的笔画通常比较清晰，140是一个稳健值
    _, thresh = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY_INV)

    # 基础清理形态学
    kernel_base = np.ones((3, 3), np.uint8)
    thresh = cv2.dilate(thresh, kernel_base, iterations=1)
    thresh = cv2.erode(thresh, kernel_base, iterations=1)

    # 轮廓提取
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    file_digit_count = 0  # 当前文件内找到的数字计数

    # --- 内层循环：遍历当前文件中的每个数字 ---
    for idx, contour in enumerate(contours):
        # 面积过滤：数字4的面积通常适中，太小的可能是噪点
        if cv2.contourArea(contour) < 40:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        digit_roi = thresh[y:y + h, x:x + w]

        # 简单的宽高比过滤：4不应该是特别扁平的
        if h < w * 0.5: continue

        # ====== 生成 4 个版本 (使用 base_name 作为文件名前缀) ======

        # v1: 基准版 (Standard)
        img_v1 = resize_and_pad(digit_roi, target_size=20)
        # 文件名格式：原文件名_编号_版本_类型.png
        save_name = f"{base_name}_id{file_digit_count}_v1_std.png"
        Image.fromarray(img_v1).save(os.path.join(save_dir, save_name))

        # v2: 旋转版 (Rotation Only)
        angle = random.uniform(-15, 15)
        img_v2 = apply_rotation(img_v1, angle)
        save_name = f"{base_name}_id{file_digit_count}_v2_rot.png"
        Image.fromarray(img_v2).save(os.path.join(save_dir, save_name))

        # v3: 形变版 (Scale + Thickness)
        rand_size = random.choice([16, 18, 22, 24])  # 避开标准20
        img_temp = resize_and_pad(digit_roi, target_size=rand_size)
        img_v3 = apply_thickness(img_temp)
        save_name = f"{base_name}_id{file_digit_count}_v3_st.png"
        Image.fromarray(img_v3).save(os.path.join(save_dir, save_name))

        # v4: 全家桶版 (Combo)
        rand_size_c = random.randint(17, 23)
        rand_angle_c = random.uniform(-18, 18)
        img_c = resize_and_pad(digit_roi, target_size=rand_size_c)
        img_c = apply_rotation(img_c, rand_angle_c)
        img_v4 = apply_thickness(img_c)
        save_name = f"{base_name}_id{file_digit_count}_v4_combo.png"
        Image.fromarray(img_v4).save(os.path.join(save_dir, save_name))

        # ========================================================
        file_digit_count += 1
        total_images_generated += 4

    print(f"  - 在 {filepath} 中找到了 {file_digit_count} 个有效的数字'4'")
    total_digits_found += file_digit_count

print(f"\n=== 全部处理完成 ===")
print(f"累计在所有文件中找到原始数字: {total_digits_found} 个")
print(f"最终生成训练集图片: {total_images_generated} 张 (4倍扩充)")
print(f"保存路径: ./{save_dir}/")