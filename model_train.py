import copy
import time
import os
from PIL import Image
import glob
import random

import torch
from torch import nn
from torchvision.datasets import FashionMNIST
import numpy as np
from torchvision import transforms
import torch.utils.data as data
import matplotlib.pyplot as plt
import pandas as pd
import torch.nn.functional as F
from model import *


# ================ 自定义MNIST数据集类 ================
class CustomMNISTDataset(data.Dataset):
    """自定义MNIST数据集类，用于从本地文件夹读取jpg/png格式的MNIST图片"""

    def __init__(self, root_dir='mnist_/train', samples_per_class=1000, transform=None):
        """
        参数:
            root_dir: 数据根目录，包含0-9子文件夹
            samples_per_class: 每个类别加载的图片数量
            transform: 数据转换
        """
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []

        # 遍历0-9数字文件夹
        for label in range(10):
            class_dir = os.path.join(root_dir, str(label))

            if not os.path.exists(class_dir):
                print(f"警告: 目录 {class_dir} 不存在，跳过...")
                continue

            # 获取所有jpg和png文件
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                image_files.extend(glob.glob(os.path.join(class_dir, ext)))

            if not image_files:
                print(f"警告: 目录 {class_dir} 中没有找到图片文件")
                continue

            # 如果文件数量不足，使用全部文件
            if len(image_files) < samples_per_class:
                print(f"注意: 数字 {label} 只有 {len(image_files)} 张图片，将使用全部")
                selected_files = image_files
            else:
                # 随机选择指定数量的图片
                selected_files = random.sample(image_files, samples_per_class)

            # 添加图片和标签
            for img_path in selected_files:
                self.images.append(img_path)
                self.labels.append(label)

        print(f"总共加载了 {len(self.images)} 张图片")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]

        try:
            # 打开图片并转换为灰度图
            img = Image.open(img_path)
            if img.mode != 'L':
                img = img.convert('L')
        except Exception as e:
            print(f"无法读取图片 {img_path}: {e}")
            # 返回一个空白图片作为占位符
            img = Image.new('L', (28, 28), color=0)

        # 应用转换
        if self.transform:
            img = self.transform(img)

        return img, label


# ================ 修改数据加载函数 ================
def train_val_data_process_local():
    """
    从本地文件夹加载MNIST图片数据
    每个类别取1000张，共10000张图片
    按照8:2划分训练集和验证集
    """
    # 定义数据转换
    transform = transforms.Compose([
        transforms.Resize(size=227),  # AlexNet需要227x227输入
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # 归一化到[-1, 1]
    ])

    # 创建自定义数据集
    full_dataset = CustomMNISTDataset(
        root_dir='mnist_/train',
        samples_per_class=1000,
        transform=transform
    )

    # 划分训练集和验证集 (80%训练, 20%验证)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_data, val_data = data.random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # 固定随机种子以便复现
    )

    # 创建数据加载器
    train_loader = data.DataLoader(
        dataset=train_data,
        batch_size=128,
        shuffle=True,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False
    )

    val_loader = data.DataLoader(
        dataset=val_data,
        batch_size=128,
        shuffle=False,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False
    )

    print(f"训练集大小: {len(train_data)}")
    print(f"验证集大小: {len(val_data)}")

    # 显示一些样本
    show_samples(train_loader)

    return train_loader, val_loader


def show_samples(data_loader, num_samples=10):
    """显示一些样本图片"""
    # 获取一个batch的数据
    images, labels = next(iter(data_loader))

    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    axes = axes.flatten()

    for i in range(min(num_samples, len(images))):
        img = images[i].numpy().squeeze()  # 去掉通道维度
        label = labels[i].item()

        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f"Label: {label}")
        axes[i].axis('off')

    plt.suptitle('MNIST_samples')
    plt.tight_layout()
    plt.show()


def train_model(model, train_loader, val_loader, epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    model = model.to(device)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    since = time.time()

    for epoch in range(epochs):
        print(f'\nEpoch {epoch + 1}/{epochs}')
        print('-' * 40)

        # 训练阶段
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        train_num = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            train_loss += loss.item() * inputs.size(0)
            train_acc += torch.sum(preds == labels.data)
            train_num += inputs.size(0)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        val_num = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)
                val_loss += loss.item() * inputs.size(0)
                val_acc += torch.sum(preds == labels.data)
                val_num += inputs.size(0)

        # 计算平均损失和准确率
        epoch_train_loss = train_loss / train_num
        epoch_train_acc = train_acc.double().item() / train_num
        epoch_val_loss = val_loss / val_num
        epoch_val_acc = val_acc.double().item() / val_num

        train_losses.append(epoch_train_loss)
        train_accs.append(epoch_train_acc)
        val_losses.append(epoch_val_loss)
        val_accs.append(epoch_val_acc)

        print(f'Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.4f}')
        print(f'Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.4f}')

        # 保存最佳模型
        if epoch_val_acc > best_acc:
            best_acc = epoch_val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            print(f'保存最佳模型，准确率: {best_acc:.4f}')

        time_elapsed = time.time() - since
        print(f'用时: {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

    # 加载最佳模型权重
    model.load_state_dict(best_model_wts)

    # 保存模型
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_acc': best_acc,
    }, './best_mnist_model.pth')

    print(f'\n训练完成！最佳验证准确率: {best_acc:.4f}')

    # 创建训练过程DataFrame
    train_process = pd.DataFrame({
        'epoch': range(1, epochs + 1),
        'train_loss': train_losses,
        'train_acc': train_accs,
        'val_loss': val_losses,
        'val_acc': val_accs,
    })

    return train_process, model


# ================ 可视化函数 ================
def matplot_acc_loss(train_process):
    plt.figure(figsize=(12, 4))

    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_process['epoch'], train_process['train_loss'], 'ro-', label='Train Loss')
    plt.plot(train_process['epoch'], train_process['val_loss'], 'bs-', label='Val Loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.grid(True, alpha=0.3)

    # 准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(train_process['epoch'], train_process['train_acc'], 'ro-', label='Train Acc')
    plt.plot(train_process['epoch'], train_process['val_acc'], 'bs-', label='Val Acc')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# ================ 主程序 ================
if __name__ == '__main__':
    print("=" * 50)
    print("MNIST图片分类训练程序")
    print("=" * 50)

    # 1. 创建模型
    print("\n1. 创建AlexNet模型...")
    model = AlexNet()

    # 打印模型结构
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"模型结构:")
    print(model)

    # 2. 加载数据
    print("\n2. 从本地文件夹加载数据...")
    print("数据目录: mnist_/train/")
    print("每个类别取1000张图片，共10000张")
    train_loader, val_loader = train_val_data_process_local()

    # 3. 训练模型
    print("\n3. 开始训练模型...")
    epochs = 20
    train_process, trained_model = train_model(model, train_loader, val_loader, epochs)

    # 4. 可视化训练过程
    print("\n4. 可视化训练过程...")
    matplot_acc_loss(train_process)

    # 6. 保存训练过程
    train_process.to_csv('./training_history.csv', index=False)
    print(f"训练历史已保存到: ./training_history.csv")
    print(f"最佳模型已保存到: ./best_mnist_model.pth")