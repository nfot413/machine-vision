import cv2
import numpy as np
from PIL import Image
import os


# 1. 创建保存文件夹
save_dir = "$_mnist"
os.makedirs(save_dir, exist_ok=True)


img = cv2.imread("$4.jpg")


if img is None:
    raise FileNotFoundError("未找到图片文件，请检查路径/URL是否正确！")

# 3. 预处理（针对$的形状调整参数）
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 二值化：$的对比度可能不同，阈值可调整（建议范围120-160）
_, thresh = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY_INV)

# 形态学操作：$有闭合区域，用3×3的核膨胀/腐蚀，增强轮廓
kernel = np.ones((3, 3), np.uint8)
thresh = cv2.dilate(thresh, kernel, iterations=1)  # 膨胀
thresh = cv2.erode(thresh, kernel, iterations=1)  # 腐蚀（去毛刺）

# 4. 分割每个$并转28×28
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
valid_count = 0

for idx, contour in enumerate(contours):
    # 过滤噪声：$的轮廓面积比V大，调大过滤阈值（比如50）
    if cv2.contourArea(contour) < 50:
        continue

    # 裁剪区域
    x, y, w, h = cv2.boundingRect(contour)
    dollar_roi = thresh[y:y + h, x:x + w]

    # 缩放并居中到28×28
    mnist_canvas = np.zeros((28, 28), dtype=np.uint8)
    scale = min(28 / w, 28 / h)
    resized_w, resized_h = int(w * scale), int(h * scale)
    resized_dollar = cv2.resize(dollar_roi, (resized_w, resized_h), interpolation=cv2.INTER_AREA)

    # 居中偏移
    x_pad = (28 - resized_w) // 2
    y_pad = (28 - resized_h) // 2
    mnist_canvas[y_pad:y_pad + resized_h, x_pad:x_pad + resized_w] = resized_dollar

    # 保存到文件夹
    save_path = os.path.join(save_dir, f"4_dollar_mnist_{valid_count}.png")
    Image.fromarray(mnist_canvas).save(save_path)
    valid_count += 1

print(f"处理完成！共生成 {valid_count} 个MNIST格式的$图片，保存至 {save_dir} 文件夹。")