import cv2
import numpy as np
from PIL import Image
import os  # 导入os模块用于文件夹操作

# 创建V_mnist文件夹（若不存在则创建，存在则不报错）
save_dir = "V_mnist"
os.makedirs(save_dir, exist_ok=True)

# 1. 读取图像并预处理
# 替换为你的图片路径（也可使用你提供的在线URL下载图片）
img = cv2.imread("V6.jpg")
if img is None:
    raise FileNotFoundError("未找到图片文件，请检查路径是否正确！")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转灰度图

# 2. 二值化（区分“V”和背景，背景→黑，V→白，阈值可调整）
_, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
kernel = np.ones((2, 2), np.uint8)
thresh = cv2.dilate(thresh, kernel, iterations=1)  # 膨胀增强轮廓

# 3. 分割每个“V”并转28×28 MNIST格式
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
valid_count = 0  # 统计有效“V”的数量

for idx, contour in enumerate(contours):
    # 过滤噪声（跳过太小的轮廓，数值可根据实际调整）
    if cv2.contourArea(contour) < 30:
        continue

    # 裁剪单个“V”的区域
    x, y, w, h = cv2.boundingRect(contour)
    v_roi = thresh[y:y + h, x:x + w]

    # 缩放到28×28并居中（保持宽高比）
    mnist_canvas = np.zeros((28, 28), dtype=np.uint8)
    scale = min(28 / w, 28 / h)  # 计算缩放比例
    resized_w, resized_h = int(w * scale), int(h * scale)
    resized_v = cv2.resize(v_roi, (resized_w, resized_h), interpolation=cv2.INTER_AREA)

    # 计算居中的偏移量
    x_pad = (28 - resized_w) // 2
    y_pad = (28 - resized_h) // 2
    mnist_canvas[y_pad:y_pad + resized_h, x_pad:x_pad + resized_w] = resized_v

    # 4. 保存到V_mnist文件夹中
    save_path = os.path.join(save_dir, f"6_v_mnist_{valid_count}.png")
    Image.fromarray(mnist_canvas).save(save_path)
    valid_count += 1

print(f"处理完成！共生成 {valid_count} 个MNIST格式的PNG文件，保存至 {save_dir} 文件夹。")