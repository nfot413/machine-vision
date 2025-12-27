import cv2
import numpy as np
from PIL import Image
import os
import random
import glob

# --- 配置区域 ---
target_number = "8"  # 目标数字
input_pattern = f"{target_number}_*.jpg"  # 匹配 8_1.jpg, 8_2.jpg...
save_dir = "eight_mnist_optimized"  # 保存路径
os.makedirs(save_dir, exist_ok=True)


# ----------------

# ==========================================
# --- 核心工具函数 ---
# ==========================================

def resize_and_pad(roi, target_size):
    """
    缩放并居中
    使用 INTER_AREA 插值，这对保留"8"中间的两个孔洞非常重要
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
    【防断裂逻辑】在原图(ROI)阶段修改粗细
    """
    if mode == 'none': return roi

    h, w = roi.shape
    # 根据原图大小动态调整核的尺寸，防止小图被过度处理
    k_size = 3 if min(h, w) > 50 else 2

    # 变粗：矩形核
    if mode == 'thicker':
        kernel = np.ones((k_size, k_size), np.uint8)
        return cv2.dilate(roi, kernel, iterations=1)

    # 变细：关键点！使用十字核 (MORPH_CROSS)
    # 十字核只腐蚀上下左右，不腐蚀对角线。
    # 这对 "8" 这种有交叉点的数字至关重要，防止中间断开。
    elif mode == 'thinner':
        cross_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (k_size, k_size))
        return cv2.erode(roi, cross_kernel, iterations=1)

    return roi


# ==========================================
# --- 主流程 ---
# ==========================================

input_files = glob.glob(input_pattern)
input_files.sort()

if not input_files:
    print(f"未找到符合 {input_pattern} 的图片！")
    exit()

print(f"找到 {len(input_files)} 个文件，开始处理数字 '8'...")
total_images = 0

for filepath in input_files:
    base_name = os.path.splitext(os.path.basename(filepath))[0]
    print(f"处理: {filepath}")

    img = cv2.imread(filepath)
    if img is None: continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 阈值：8的交叉点容易黑成一片，140通常比较稳
    _, thresh = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY_INV)

    # 基础修复：先闭合断点，再清理毛刺
    kernel_base = np.ones((3, 3), np.uint8)
    thresh = cv2.dilate(thresh, kernel_base, iterations=1)
    thresh = cv2.erode(thresh, kernel_base, iterations=1)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    file_digit_cnt = 0

    for contour in contours:
        # 面积过滤
        if cv2.contourArea(contour) < 40: continue

        x, y, w, h = cv2.boundingRect(contour)

        # 形状过滤：8是竖直的。如果太扁 (h < w * 0.7)，可能是躺着的 ∞ 或者噪点
        if h < w * 0.7: continue

        # 获取高清原图 ROI
        roi_original = thresh[y:y + h, x:x + w]

        # ====== 4 倍生成策略 ======

        # 1. 【基准版】
        img_v1 = resize_and_pad(roi_original, target_size=20)
        Image.fromarray(img_v1).save(os.path.join(save_dir, f"{base_name}_{file_digit_cnt}_v1_std.png"))

        # 2. 【旋转版】
        angle = random.uniform(-15, 15)
        img_v2 = apply_rotation(img_v1, angle)
        Image.fromarray(img_v2).save(os.path.join(save_dir, f"{base_name}_{file_digit_cnt}_v2_rot.png"))

        # 3. 【形变版】 (先变粗细 -> 后缩放)
        thick_mode = random.choice(['thicker', 'thinner'])
        roi_mod = pre_process_thickness(roi_original, thick_mode)

        # 8如果太小，洞会看不清，所以随机尺寸稍微偏大一点点
        rand_size = random.choice([18, 20, 22, 24])
        img_v3 = resize_and_pad(roi_mod, target_size=rand_size)
        Image.fromarray(img_v3).save(os.path.join(save_dir, f"{base_name}_{file_digit_cnt}_v3_st.png"))

        # 4. 【全家桶】 (先变粗细 -> 后缩放 -> 后旋转)
        thick_mode_c = random.choice(['thicker', 'thinner', 'none'])
        roi_c = pre_process_thickness(roi_original, thick_mode_c)

        rand_size_c = random.randint(17, 23)
        img_c = resize_and_pad(roi_c, target_size=rand_size_c)

        rand_angle_c = random.uniform(-15, 15)
        img_v4 = apply_rotation(img_c, rand_angle_c)

        Image.fromarray(img_v4).save(os.path.join(save_dir, f"{base_name}_{file_digit_cnt}_v4_combo.png"))

        file_digit_cnt += 1
        total_images += 4

    print(f"  -> 提取出 {file_digit_cnt} 个 '8'")

print(f"\n全部完成！共生成 {total_images} 张图片。")
print(f"保存路径: {save_dir}")