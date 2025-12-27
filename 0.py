import cv2
import numpy as np
from PIL import Image
import os

# --- 配置区域 ---
input_filename = "0_5.jpg"  # 请确保你的目录下有这张包含数字0的图片
save_dir = "zero_mnist"  # 保存结果的文件夹
# ----------------

# 1. 创建保存文件夹
os.makedirs(save_dir, exist_ok=True)

# 2. 读取图片
img = cv2.imread(input_filename)

# 检查图片是否读取成功
if img is None:
    raise FileNotFoundError(f"未找到图片文件 '{input_filename}'，请检查文件名或路径是否正确！")

# 3. 图像预处理
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转灰度图

# 二值化：背景黑，数字白
# 阈值130是经验值。如果你的图片光线较暗，可能需要调低（如100）；如果很亮，调高（如150）
# 也可以使用 cv2.THRESH_OTSU 自动寻找阈值
_, thresh = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY_INV)

# 形态学操作：关键步骤
kernel = np.ones((3, 3), np.uint8)

# 【修改点】：对于数字0，膨胀次数改为1。
# 如果膨胀太多(比如2或3)，0中间的洞可能会被填满，导致变成实心圆。
thresh = cv2.dilate(thresh, kernel, iterations=1)
thresh = cv2.erode(thresh, kernel, iterations=1)  # 腐蚀去除边缘毛刺

# 4. 分割每个"0"并转换为28×28 MNIST格式
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
valid_count = 0

print(f"检测到 {len(contours)} 个轮廓，开始处理...")

for idx, contour in enumerate(contours):
    # 过滤噪声：跳过面积太小的噪点
    # 如果你的0写得很小，可以将60调小
    if cv2.contourArea(contour) < 60:
        continue

    # 裁剪单个数字的区域
    x, y, w, h = cv2.boundingRect(contour)
    digit_roi = thresh[y:y + h, x:x + w]

    # --- MNIST 标准化处理 (核心逻辑) ---

    # 创建黑色画布 28x28
    mnist_canvas = np.zeros((28, 28), dtype=np.uint8)

    # 计算缩放比例：将数字缩放到最大边为20-24左右（保留一点边距，MNIST通常数字主体占20x20）
    # 这里我们缩放到最大边长为28的 "紧凑填充"，如果想要更标准的MNIST，建议改为 20/w 或 20/h
    scale = min(28 / w, 28 / h)

    resized_w, resized_h = int(w * scale), int(h * scale)

    # 避免缩放后尺寸为0的极端情况
    if resized_w == 0 or resized_h == 0:
        continue

    # 用INTER_CUBIC插值：更适合曲线的缩放
    resized_digit = cv2.resize(digit_roi, (resized_w, resized_h), interpolation=cv2.INTER_CUBIC)

    # 计算居中偏移量
    x_pad = (28 - resized_w) // 2
    y_pad = (28 - resized_h) // 2

    # 将缩放后的数字放入画布中心
    mnist_canvas[y_pad:y_pad + resized_h, x_pad:x_pad + resized_w] = resized_digit

    # 5. 保存结果
    # 标签设为0 (对应MNIST的label)
    save_path = os.path.join(save_dir, f"5_0_mnist_{valid_count}.png")
    Image.fromarray(mnist_canvas).save(save_path)
    valid_count += 1

print(f"--- 处理完成 ---")
print(f"共生成 {valid_count} 个MNIST格式的'0'图片")
print(f"保存位置: ./{save_dir}/")