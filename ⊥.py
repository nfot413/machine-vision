import cv2
import numpy as np
from PIL import Image
import os


# 1. 创建保存文件夹（用perp_mnist避免特殊字符问题）
save_dir = "perp_mnist"
os.makedirs(save_dir, exist_ok=True)


img = cv2.imread("perp4.jpg")

# 检查图片是否读取成功
if img is None:
    raise FileNotFoundError("未找到图片文件，请检查路径/URL是否正确！")

# 3. 图像预处理（适配⊥的直线特征）
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转灰度图

# 二值化：背景黑，⊥白（阈值135，可根据图片亮度调整120-150）
_, thresh = cv2.threshold(gray, 135, 255, cv2.THRESH_BINARY_INV)

# 形态学操作：3×3核膨胀+腐蚀，强化⊥的直线轮廓，去除毛刺
kernel = np.ones((3, 3), np.uint8)
thresh = cv2.dilate(thresh, kernel, iterations=1)  # 膨胀让轮廓更粗
thresh = cv2.erode(thresh, kernel, iterations=1)  # 腐蚀去除边缘噪声

# 4. 分割每个⊥并转换为28×28 MNIST格式
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
valid_count = 0  # 统计有效⊥的数量

for idx, contour in enumerate(contours):
    # 过滤噪声：跳过面积小于40的轮廓（⊥的轮廓更大）
    if cv2.contourArea(contour) < 40:
        continue

    # 裁剪单个⊥的区域
    x, y, w, h = cv2.boundingRect(contour)
    perp_roi = thresh[y:y + h, x:x + w]

    # 缩放并居中到28×28画布（保持宽高比）
    mnist_canvas = np.zeros((28, 28), dtype=np.uint8)
    scale = min(28 / w, 28 / h)  # 计算缩放比例
    resized_w, resized_h = int(w * scale), int(h * scale)
    resized_perp = cv2.resize(perp_roi, (resized_w, resized_h), interpolation=cv2.INTER_AREA)

    # 计算居中偏移量
    x_pad = (28 - resized_w) // 2
    y_pad = (28 - resized_h) // 2
    mnist_canvas[y_pad:y_pad + resized_h, x_pad:x_pad + resized_w] = resized_perp

    # 5. 保存到perp_mnist文件夹
    save_path = os.path.join(save_dir, f"4_perp_mnist_{valid_count}.png")
    Image.fromarray(mnist_canvas).save(save_path)
    valid_count += 1

print(f"处理完成！共生成 {valid_count} 个MNIST格式的⊥图片，保存至 {save_dir} 文件夹。")