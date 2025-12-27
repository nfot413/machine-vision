import cv2
import numpy as np
from PIL import Image
import os
import random

# --- 配置区域 ---
input_filename = "2.jpg"
save_dir = "two_mnist_x4_combo"  # 保存目录
os.makedirs(save_dir, exist_ok=True)


# ----------------

# --- 工具函数 ---

def resize_and_pad(roi, target_size):
    """
    缩放并居中
    target_size: 数字长边的像素大小 (MNIST标准约20)
    """
    h, w = roi.shape
    canvas = np.zeros((28, 28), dtype=np.uint8)

    # 避免除以0
    if w == 0 or h == 0: return canvas

    scale = min(target_size / w, target_size / h)
    new_w, new_h = int(w * scale), int(h * scale)
    new_w, new_h = max(1, new_w), max(1, new_h)

    # 使用 INTER_AREA 插值，线条更干净
    resized = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_AREA)

    x_pad = (28 - new_w) // 2
    y_pad = (28 - new_h) // 2

    # 边界保护
    h_end = min(y_pad + new_h, 28)
    w_end = min(x_pad + new_w, 28)

    canvas[y_pad:h_end, x_pad:w_end] = resized[:h_end - y_pad, :w_end - x_pad]
    return canvas


def apply_rotation(image, angle):
    """
    旋转图片 (背景保持黑色)
    """
    h, w = image.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return rotated


def apply_thickness(image):
    """
    随机变粗或变细
    """
    mode = random.choice(['thicker', 'thinner', 'none'])  # 增加'none'保持原样的情况
    kernel = np.ones((2, 2), np.uint8)  # 2x2核，细腻控制

    if mode == 'thicker':
        return cv2.dilate(image, kernel, iterations=1)
    elif mode == 'thinner':
        return cv2.erode(image, kernel, iterations=1)
    else:
        return image


# --- 主程序 ---

img = cv2.imread(input_filename)
if img is None:
    raise FileNotFoundError("未找到图片！")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY_INV)

# 基础清理
kernel_base = np.ones((3, 3), np.uint8)
thresh = cv2.dilate(thresh, kernel_base, iterations=1)
thresh = cv2.erode(thresh, kernel_base, iterations=1)

contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
valid_count = 0
total_files = 0

print(f"检测到 {len(contours)} 个轮廓，开始生成 4倍 组合数据...")

for idx, contour in enumerate(contours):
    if cv2.contourArea(contour) < 40:
        continue

    x, y, w, h = cv2.boundingRect(contour)
    digit_roi = thresh[y:y + h, x:x + w]

    # ====== 生成 4 个版本 ======

    # 1. 【基准版】 (Standard)
    # 也就是最标准的MNIST样子：20px大小，正中间
    img_v1 = resize_and_pad(digit_roi, target_size=20)

    save_path = os.path.join(save_dir, f"2_{valid_count}_v1_std.png")
    Image.fromarray(img_v1).save(save_path)
    total_files += 1

    # 2. 【旋转版】 (Rotation Only)
    # 标准大小 + 随机旋转
    angle = random.uniform(-15, 15)
    img_v2 = apply_rotation(img_v1, angle)  # 直接复用v1进行旋转

    save_path = os.path.join(save_dir, f"2_{valid_count}_v2_rot.png")
    Image.fromarray(img_v2).save(save_path)
    total_files += 1

    # 3. 【形变版】 (Scale + Thickness)
    # 随机大小 (15-24px) + 随机粗细
    rand_size = random.choice([15, 16, 22, 23, 24])  # 避开标准的20
    img_temp = resize_and_pad(digit_roi, target_size=rand_size)
    img_v3 = apply_thickness(img_temp)

    save_path = os.path.join(save_dir, f"2_{valid_count}_v3_st.png")  # st = scale + thickness
    Image.fromarray(img_v3).save(save_path)
    total_files += 1

    # 4. 【全家桶版】 (Combo: Scale + Rotation + Thickness)
    # 三种效果全部随机叠加
    rand_size_combo = random.randint(16, 22)  # 大小随机
    rand_angle_combo = random.uniform(-18, 18)  # 角度随机(稍微加大一点范围)

    # 步骤：先缩放 -> 再旋转 -> 最后调粗细
    img_c = resize_and_pad(digit_roi, target_size=rand_size_combo)
    img_c = apply_rotation(img_c, rand_angle_combo)
    img_v4 = apply_thickness(img_c)

    save_path = os.path.join(save_dir, f"2_{valid_count}_v4_combo.png")
    Image.fromarray(img_v4).save(save_path)
    total_files += 1

    # ==========================
    valid_count += 1

print(f"--- 处理完成 ---")
print(f"原始数字: {valid_count} 个")
print(f"最终产出: {total_files} 张图片 (4倍扩充)")
print(f"保存路径: {save_dir}")