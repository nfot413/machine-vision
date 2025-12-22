import cv2
import numpy as np
from PIL import Image
import os
import requests  # 用于在线图片读取

# 1. 创建保存文件夹（用theta_mnist避免特殊字符问题）
save_dir = "theta_mnist"
os.makedirs(save_dir, exist_ok=True)

img = cv2.imread("theta6.jpg")

# 检查图片是否读取成功
if img is None:
    raise FileNotFoundError("未找到图片文件，请检查路径/URL是否正确！")

# 3. 图像预处理（适配θ的曲线特征）
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转灰度图

# 二值化：背景黑，θ白（阈值130，可根据图片亮度调整120-140）
_, thresh = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY_INV)

# 形态学操作：强化θ的圆弧轮廓，防止断裂
kernel = np.ones((3, 3), np.uint8)
thresh = cv2.dilate(thresh, kernel, iterations=2)  # 两次膨胀：让圆弧更连续
thresh = cv2.erode(thresh, kernel, iterations=1)  # 一次腐蚀：去除边缘噪声

# 4. 分割每个θ并转换为28×28 MNIST格式
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
valid_count = 0  # 统计有效θ的数量

for idx, contour in enumerate(contours):
    # 过滤噪声：跳过面积小于60的轮廓（θ的轮廓更大）
    if cv2.contourArea(contour) < 60:
        continue

    # 裁剪单个θ的区域
    x, y, w, h = cv2.boundingRect(contour)
    theta_roi = thresh[y:y + h, x:x + w]

    # 缩放并居中到28×28画布（保持宽高比）
    mnist_canvas = np.zeros((28, 28), dtype=np.uint8)
    scale = min(28 / w, 28 / h)  # 计算缩放比例
    resized_w, resized_h = int(w * scale), int(h * scale)
    # 用INTER_CUBIC插值：更适合曲线的缩放，避免锯齿
    resized_theta = cv2.resize(theta_roi, (resized_w, resized_h), interpolation=cv2.INTER_CUBIC)

    # 计算居中偏移量
    x_pad = (28 - resized_w) // 2
    y_pad = (28 - resized_h) // 2
    mnist_canvas[y_pad:y_pad + resized_h, x_pad:x_pad + resized_w] = resized_theta

    # 5. 保存到theta_mnist文件夹
    save_path = os.path.join(save_dir, f"6_theta_mnist_{valid_count}.png")
    Image.fromarray(mnist_canvas).save(save_path)
    valid_count += 1

print(f"处理完成！共生成 {valid_count} 个MNIST格式的θ图片，保存至 {save_dir} 文件夹。")