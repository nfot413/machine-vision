import cv2
import numpy as np
from PIL import Image
import os

# --- 配置区域 ---
input_filename = "1_3.jpg"
save_dir = "one_mnist_thin"  # 改个文件夹名，方便区分
os.makedirs(save_dir, exist_ok=True)
# ----------------

img = cv2.imread(input_filename)
if img is None:
    raise FileNotFoundError("未找到图片文件！")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 【关键修改点 1】：提高阈值
# 原来是130，现在改为 160 或更高。
# 原理：阈值越高，只有最黑的笔迹会被保留，笔迹边缘的灰色过渡会被切掉，线条自然变细。
_, thresh = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY_INV)

# 定义形态学核
kernel = np.ones((3, 3), np.uint8)

# 【关键修改点 2】：取消膨胀 (Dilate)
# 之前的代码这里是 iterations=2，直接注释掉！
# thresh = cv2.dilate(thresh, kernel, iterations=1)

# 【关键修改点 3】：增加腐蚀 (Erode) —— 瘦身操作
# 如果出来的结果还不够细，将 iterations 改为 1。
# 如果太细断开了，就改为 0。
thresh = cv2.erode(thresh, kernel, iterations=1)

# 4. 分割与标准化
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
valid_count = 0

print(f"检测到 {len(contours)} 个轮廓...")

for idx, contour in enumerate(contours):
    # 这里的面积阈值保持较小，因为变细后面积会更小
    if cv2.contourArea(contour) < 15:
        continue

    x, y, w, h = cv2.boundingRect(contour)

    # 过滤掉扁平的噪点 (1应该是瘦高的)
    if w > h:
        continue

    digit_roi = thresh[y:y + h, x:x + w]

    # --- MNIST 标准化 ---
    mnist_canvas = np.zeros((28, 28), dtype=np.uint8)

    # 计算缩放
    scale = min(28 / w, 28 / h)
    resized_w, resized_h = int(w * scale), int(h * scale)
    resized_w, resized_h = max(1, resized_w), max(1, resized_h)

    # 【微调】：使用 INTER_AREA 插值
    # INTER_CUBIC 可能会产生波纹让线条显粗，INTER_AREA 更适合缩小图像，线条更锐利
    resized_digit = cv2.resize(digit_roi, (resized_w, resized_h), interpolation=cv2.INTER_AREA)

    # 居中
    x_pad = (28 - resized_w) // 2
    y_pad = (28 - resized_h) // 2

    # 赋值
    mnist_canvas[y_pad:y_pad + resized_h, x_pad:x_pad + resized_w] = resized_digit

    # 保存
    save_path = os.path.join(save_dir, f"3_1_mnist_thin_{valid_count}.png")
    Image.fromarray(mnist_canvas).save(save_path)
    valid_count += 1

print(f"--- 处理完成 ---")
print(f"更细的'1'已保存至 {save_dir}，共 {valid_count} 张。")