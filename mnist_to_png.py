import os
import numpy as np
from PIL import Image
from tensorflow.keras.datasets import mnist

# 1. 加载MNIST数据集
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 2. 定义保存路径（建议创建分级目录：根目录/训练集/标签/图片.jpg）
root_dir = "mnist"
train_dir = os.path.join(root_dir, "train")
test_dir = os.path.join(root_dir, "test")

# 3. 创建目录（如果不存在）
for dir_path in [root_dir, train_dir, test_dir]:
    os.makedirs(dir_path, exist_ok=True)
# 为每个标签创建子目录（0-9）
for label in range(10):
    os.makedirs(os.path.join(train_dir, str(label)), exist_ok=True)
    os.makedirs(os.path.join(test_dir, str(label)), exist_ok=True)

# 4. 保存训练集为JPG
for idx, (img_array, label) in enumerate(zip(x_train, y_train)):
    # 转换为PIL图像（MNIST数组是uint8类型，直接可用）
    img = Image.fromarray(img_array, mode="L")  # mode=L表示灰度图
    # 定义文件名：train_索引_标签.jpg
    img_name = f"train_{idx}_{label}.jpg"
    # 保存路径：train/标签/文件名
    save_path = os.path.join(train_dir, str(label), img_name)
    img.save(save_path)
    # 可选：打印进度（每1000张打印一次）
    if idx % 1000 == 0:
        print(f"已保存训练集 {idx} 张图片")

# 5. 保存测试集为JPG
for idx, (img_array, label) in enumerate(zip(x_test, y_test)):
    img = Image.fromarray(img_array, mode="L")
    img_name = f"test_{idx}_{label}.jpg"
    save_path = os.path.join(test_dir, str(label), img_name)
    img.save(save_path)
    if idx % 1000 == 0:
        print(f"已保存测试集 {idx} 张图片")

print("MNIST数据集已全部保存为JPG格式！")