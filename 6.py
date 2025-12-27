import cv2
import numpy as np
from PIL import Image
import os
import random
import glob

# --- 配置区域 ---
target_number = "6"  # 这里设置你要处理的数字
input_pattern = f"{target_number}_*.jpg"  # 自动匹配 6_1.jpg, 6_2.jpg...
save_dir = "six_mnist_optimized"  # 保存目录
os.makedirs(save_dir, exist_ok=True)


# ----------------

# ==========================================
# --- 核心工具函数 (逻辑已升级) ---
# ==========================================

def resize_and_pad(roi, target_size):
    """
    缩放并居中 (使用 INTER_AREA 插值，让线条更平滑)
    """
    h, w = roi.shape
    canvas = np.zeros((28, 28), dtype=np.uint8)
    if w == 0 or h == 0: return canvas

    scale = min(target_size / w, target_size / h)
    new_w, new_h = int(w * scale), int(h * scale)
    new_w, new_h = max(1, new_w), max(1, new_h)

    resized = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_AREA)

    x_pad = (28 - new_w) // 2
    y_pad = (28 - new_h) // 2

    h_end = min(y_pad + new_h, 28)
    w_end = min(x_pad + new_w, 28)

    canvas[y_pad:h_end, x_pad:w_end] = resized[:h_end - y_pad, :w_end - x_pad]
    return canvas


def apply_rotation(image, angle):
    """旋转"""
    h, w = image.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return rotated


def pre_process_thickness(roi, mode):
    """
    【关键修改】在 ROI (大图) 阶段调整粗细，而不是缩小后
    mode: 'thicker', 'thinner', 'none'
    """
    if mode == 'none':
        return roi

    # 动态计算核的大小：如果图很大，核就大一点；图小，核就小一点
    h, w = roi.shape
    avg_dim = (h + w) / 2

    # 基础核大小
    k_size = 3
    if avg_dim < 50: k_size = 2  # 如果切出来的本来就很小，用小核

    kernel = np.ones((k_size, k_size), np.uint8)

    if mode == 'thicker':
        return cv2.dilate(roi, kernel, iterations=1)
    elif mode == 'thinner':
        # 即使在大图上，腐蚀也要小心，使用十字形核比矩形核更温和
        # 十字核不会腐蚀拐角，只腐蚀边缘
        cross_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (k_size, k_size))
        return cv2.erode(roi, cross_kernel, iterations=1)

    return roi


# ==========================================
# --- 主流程 ---
# ==========================================

input_files = glob.glob(input_pattern)
input_files.sort()

if not input_files:
    print(f"未找到符合 {input_pattern} 的图片。")
    exit()

print(f"找到 {len(input_files)} 个文件，开始处理 '{target_number}'...")
total_images = 0

for filepath in input_files:
    base_name = os.path.splitext(os.path.basename(filepath))[0]
    print(f"处理: {filepath}")

    img = cv2.imread(filepath)
    if img is None: continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 阈值：如果是6，可以用140；如果线条细，可以稍微调高到150
    _, thresh = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY_INV)

    # 基础降噪
    kernel_base = np.ones((3, 3), np.uint8)
    thresh = cv2.dilate(thresh, kernel_base, iterations=1)
    thresh = cv2.erode(thresh, kernel_base, iterations=1)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    file_cnt = 0

    for contour in contours:
        if cv2.contourArea(contour) < 40: continue  # 面积过滤
        x, y, w, h = cv2.boundingRect(contour)

        # 宽高比过滤：过滤掉太扁的噪点 (6是瘦高的)
        if h < w * 0.6: continue

        # 原始的高清 ROI
        roi_original = thresh[y:y + h, x:x + w]

        # ====== 4倍扩充 (逻辑顺序变更) ======

        # 1. 【基准版】
        # 直接缩小
        img_v1 = resize_and_pad(roi_original, target_size=20)
        Image.fromarray(img_v1).save(os.path.join(save_dir, f"{base_name}_{file_cnt}_v1_std.png"))

        # 2. 【旋转版】
        # 基准 -> 旋转
        angle = random.uniform(-15, 15)
        img_v2 = apply_rotation(img_v1, angle)
        Image.fromarray(img_v2).save(os.path.join(save_dir, f"{base_name}_{file_cnt}_v2_rot.png"))

        # 3. 【形变版】 (先变粗细 -> 再缩小)
        # 随机选择一种粗细模式
        thick_mode = random.choice(['thicker', 'thinner'])
        # 步骤A：在原图上改粗细 (这样变细不会断)
        roi_modified = pre_process_thickness(roi_original, thick_mode)
        # 步骤B：缩放到随机大小
        rand_size = random.choice([16, 17, 22, 23])
        img_v3 = resize_and_pad(roi_modified, target_size=rand_size)

        Image.fromarray(img_v3).save(os.path.join(save_dir, f"{base_name}_{file_cnt}_v3_st.png"))

        # 4. 【全家桶】 (先变粗细 -> 再缩小 -> 再旋转)
        thick_mode_c = random.choice(['thicker', 'thinner', 'none'])
        rand_size_c = random.randint(17, 22)
        rand_angle_c = random.uniform(-15, 15)

        # 顺序很重要：先处理原图，再resize，最后旋转
        roi_c = pre_process_thickness(roi_original, thick_mode_c)
        img_c = resize_and_pad(roi_c, target_size=rand_size_c)
        img_v4 = apply_rotation(img_c, rand_angle_c)

        Image.fromarray(img_v4).save(os.path.join(save_dir, f"{base_name}_{file_cnt}_v4_combo.png"))

        file_cnt += 1
        total_images += 4

    print(f"  -> {file_cnt} 个有效数字")

print(f"\n全部完成！共生成 {total_images} 张图片。")