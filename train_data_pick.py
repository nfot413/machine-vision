import os
import shutil
import random
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


class MNISTSplitter:
    """MNIST数据集分割器 - 创建训练集和测试集"""

    def __init__(self, source_path="./mnist/train"):
        """
        初始化MNIST分割器

        Args:
            source_path: 原始MNIST训练集路径，包含0-9子文件夹
        """
        self.source_path = Path(source_path)
        self.train_path = Path("./mnist_train_1500")
        self.test_path = Path("./mnist_test_500")
        self.classes = [str(i) for i in range(10)]

    def check_structure(self):
        """检查数据集的文件夹结构"""
        print("=" * 60)
        print("检查数据集结构...")
        print("=" * 60)

        if not self.source_path.exists():
            print(f"❌ 错误: 源路径不存在: {self.source_path}")
            return False

        print(f"源目录: {self.source_path}")
        print("查找类别文件夹:")

        missing_classes = []
        available_classes = []

        for class_name in self.classes:
            class_path = self.source_path / class_name
            if class_path.exists() and class_path.is_dir():
                # 统计图片数量
                image_files = list(class_path.glob("*.png")) + list(class_path.glob("*.jpg")) + \
                              list(class_path.glob("*.jpeg")) + list(class_path.glob("*.bmp"))
                count = len(image_files)
                available_classes.append(class_name)
                print(f"  ✅ {class_name}: {count} 张图片")
            else:
                missing_classes.append(class_name)
                print(f"  ❌ {class_name}: 文件夹不存在")

        if missing_classes:
            print(f"\n警告: 缺少 {len(missing_classes)} 个类别的文件夹")

        return len(available_classes) > 0

    def split_images(self, train_per_class=1500, test_per_class=500, random_seed=42):
        """
        从每个类别中随机分割图片为训练集和测试集

        Args:
            train_per_class: 每个类别的训练集数量
            test_per_class: 每个类别的测试集数量
            random_seed: 随机种子，确保可重复性
        """
        random.seed(random_seed)

        print("\n" + "=" * 60)
        print("开始分割图片...")
        print("=" * 60)
        print(
            f"每个类别: {train_per_class} 张训练集 + {test_per_class} 张测试集 = 总共 {train_per_class + test_per_class} 张")

        # 创建输出目录结构
        self.train_path.mkdir(parents=True, exist_ok=True)
        self.test_path.mkdir(parents=True, exist_ok=True)

        for class_name in self.classes:
            (self.train_path / class_name).mkdir(exist_ok=True)
            (self.test_path / class_name).mkdir(exist_ok=True)

        train_counts = {class_name: 0 for class_name in self.classes}
        test_counts = {class_name: 0 for class_name in self.classes}

        # 遍历每个类别
        for class_name in self.classes:
            class_path = self.source_path / class_name

            if not class_path.exists():
                print(f"跳过类别 {class_name}: 文件夹不存在")
                continue

            # 获取所有图片文件
            image_files = []
            for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp']:
                image_files.extend(class_path.glob(ext))

            if not image_files:
                print(f"跳过类别 {class_name}: 没有找到图片文件")
                continue

            print(f"\n处理类别 {class_name}:")
            print(f"  找到 {len(image_files)} 张图片")

            # 检查是否足够所需数量
            total_needed = train_per_class + test_per_class
            if len(image_files) < total_needed:
                print(f"  警告: 只有 {len(image_files)} 张图片，不足 {total_needed} 张")
                # 按比例分配
                train_count = int(len(image_files) * train_per_class / total_needed)
                test_count = len(image_files) - train_count
                print(f"  重新分配: {train_count} 张训练集 + {test_count} 张测试集")
            else:
                train_count = train_per_class
                test_count = test_per_class

            # 随机打乱所有图片
            random.shuffle(image_files)

            # 分割为训练集和测试集
            train_files = image_files[:train_count]
            test_files = image_files[train_count:train_count + test_count]

            # 复制训练集图片
            for i, img_file in enumerate(train_files):
                try:
                    # 生成新文件名
                    new_filename = f"{class_name}_train_{i + 1:04d}{img_file.suffix}"
                    output_file = self.train_path / class_name / new_filename
                    # 复制文件
                    shutil.copy2(img_file, output_file)
                    train_counts[class_name] += 1
                except Exception as e:
                    print(f"  错误: 复制训练集文件 {img_file.name} 失败: {e}")

            # 复制测试集图片
            for i, img_file in enumerate(test_files):
                try:
                    # 生成新文件名
                    new_filename = f"{class_name}_test_{i + 1:04d}{img_file.suffix}"
                    output_file = self.test_path / class_name / new_filename
                    # 复制文件
                    shutil.copy2(img_file, output_file)
                    test_counts[class_name] += 1
                except Exception as e:
                    print(f"  错误: 复制测试集文件 {img_file.name} 失败: {e}")

            print(f"  成功分割: {train_counts[class_name]} 张训练集 + {test_counts[class_name]} 张测试集")

        return train_counts, test_counts

    def verify_split(self, expected_train=1500, expected_test=500):
        """验证分割的数据集"""
        print("\n" + "=" * 60)
        print("验证分割的数据集...")
        print("=" * 60)

        train_results = {}
        test_results = {}
        total_train = 0
        total_test = 0

        print("训练集统计:")
        print("-" * 30)
        for class_name in self.classes:
            class_path = self.train_path / class_name

            if not class_path.exists():
                print(f"❌ 训练集类别 {class_name}: 文件夹不存在")
                train_results[class_name] = 0
                continue

            # 统计图片数量
            image_count = 0
            for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp']:
                image_count += len(list(class_path.glob(ext)))

            train_results[class_name] = image_count
            total_train += image_count

            status = "✅" if image_count == expected_train else "⚠️"
            print(f"{status} 训练集类别 {class_name}: {image_count} 张图片", end="")

            if image_count != expected_train:
                print(f" (期望: {expected_train})")
            else:
                print()

        print(f"\n训练集总计: {total_train} 张图片")
        print(f"期望训练集总计: {expected_train * 10} 张图片")

        print("\n测试集统计:")
        print("-" * 30)
        for class_name in self.classes:
            class_path = self.test_path / class_name

            if not class_path.exists():
                print(f"❌ 测试集类别 {class_name}: 文件夹不存在")
                test_results[class_name] = 0
                continue

            # 统计图片数量
            image_count = 0
            for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp']:
                image_count += len(list(class_path.glob(ext)))

            test_results[class_name] = image_count
            total_test += image_count

            status = "✅" if image_count == expected_test else "⚠️"
            print(f"{status} 测试集类别 {class_name}: {image_count} 张图片", end="")

            if image_count != expected_test:
                print(f" (期望: {expected_test})")
            else:
                print()

        print(f"\n测试集总计: {total_test} 张图片")
        print(f"期望测试集总计: {expected_test * 10} 张图片")
        print("-" * 40)
        print(f"数据集总计: {total_train + total_test} 张图片")
        print(f"期望总计: {(expected_train + expected_test) * 10} 张图片")

        return train_results, test_results

    def visualize_samples(self, samples_per_class=5):
        """可视化每个类别的样本（训练集和测试集）"""
        print("\n" + "=" * 60)
        print("可视化样本...")
        print("=" * 60)

        # 创建两个子图：一个用于训练集，一个用于测试集
        fig, axes = plt.subplots(2, 10, figsize=(20, 6))

        for class_idx, class_name in enumerate(self.classes):
            # 训练集样本
            train_class_path = self.train_path / class_name
            if train_class_path.exists():
                train_image_files = list(train_class_path.glob("*.png")) + list(train_class_path.glob("*.jpg"))
                if train_image_files:
                    # 显示训练集的第一张图片
                    ax_train = axes[0, class_idx]
                    try:
                        img = Image.open(train_image_files[0])
                        ax_train.imshow(img, cmap='gray')
                        ax_train.set_title(f"训练集 {class_name}")
                    except:
                        pass
                    ax_train.axis('off')

            # 测试集样本
            test_class_path = self.test_path / class_name
            if test_class_path.exists():
                test_image_files = list(test_class_path.glob("*.png")) + list(test_class_path.glob("*.jpg"))
                if test_image_files:
                    # 显示测试集的第一张图片
                    ax_test = axes[1, class_idx]
                    try:
                        img = Image.open(test_image_files[0])
                        ax_test.imshow(img, cmap='gray')
                        ax_test.set_title(f"测试集 {class_name}")
                    except:
                        pass
                    ax_test.axis('off')

        plt.suptitle("MNIST数据集样本分割 (每类显示1张)", fontsize=16)
        plt.tight_layout()
        plt.show()

    def create_dataset_info(self):
        """创建数据集信息文件"""
        print("\n" + "=" * 60)
        print("创建数据集信息文件...")
        print("=" * 60)

        # 训练集信息文件
        train_info_file = self.train_path / "dataset_info.txt"
        train_csv_file = self.train_path / "dataset.csv"

        # 测试集信息文件
        test_info_file = self.test_path / "dataset_info.txt"
        test_csv_file = self.test_path / "dataset.csv"

        # 创建训练集信息文件
        self._create_single_dataset_info(train_info_file, train_csv_file, self.train_path, "训练集")

        # 创建测试集信息文件
        self._create_single_dataset_info(test_info_file, test_csv_file, self.test_path, "测试集")

        # 创建总体信息文件
        overall_info_file = Path("./dataset_split_info.txt")
        with open(overall_info_file, 'w', encoding='utf-8') as f:
            f.write("MNIST数据集分割信息\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"源目录: {self.source_path}\n")
            f.write(f"训练集目录: {self.train_path}\n")
            f.write(f"测试集目录: {self.test_path}\n\n")

            f.write("分割比例:\n")
            f.write("-" * 30 + "\n")
            f.write("每个类别: 1500张训练集 + 500张测试集\n")
            f.write("总计: 15000张训练集 + 5000张测试集 = 20000张图片\n\n")

            f.write(f"创建时间: {os.path.getctime(self.train_path)}\n")

        print(f"✅ 总体信息文件已保存到: {overall_info_file}")

    def _create_single_dataset_info(self, info_file, csv_file, dataset_path, dataset_name):
        """创建单个数据集的信息文件"""
        with open(info_file, 'w', encoding='utf-8') as f:
            f.write(f"MNIST {dataset_name}信息\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"创建时间: {os.path.getctime(dataset_path)}\n")
            f.write(f"源目录: {self.source_path}\n")
            f.write(f"数据集目录: {dataset_path}\n\n")

            f.write("类别统计:\n")
            f.write("-" * 30 + "\n")

            total_images = 0
            for class_name in self.classes:
                class_path = dataset_path / class_name
                if class_path.exists():
                    image_count = 0
                    for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp']:
                        image_count += len(list(class_path.glob(ext)))

                    f.write(f"类别 {class_name}: {image_count} 张图片\n")
                    total_images += image_count

            f.write("\n")
            f.write(f"总计: {total_images} 张图片\n")
            f.write(f"平均每类: {total_images / 10 if total_images > 0 else 0} 张图片\n")

        # 创建CSV文件
        try:
            with open(csv_file, 'w', encoding='utf-8') as f:
                f.write("filename,label,path\n")
                for class_name in self.classes:
                    class_path = dataset_path / class_name
                    if class_path.exists():
                        for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp']:
                            for img_file in class_path.glob(ext):
                                f.write(f"{img_file.name},{class_name},{class_name}/{img_file.name}\n")
            print(f"✅ {dataset_name}信息已保存到: {info_file}")
            print(f"✅ {dataset_name}CSV文件已保存到: {csv_file}")
        except Exception as e:
            print(f"⚠️ 创建{dataset_name}CSV文件时出错: {e}")

    def run(self, train_per_class=1500, test_per_class=500):
        """运行完整的分割流程"""
        print("🚀 MNIST数据集分割器")
        print("📁 从每个类别中随机分割图片为训练集和测试集")
        print("=" * 60)

        # 1. 检查数据集结构
        if not self.check_structure():
            print("\n❌ 无法继续，请检查数据集结构！")
            return

        # 2. 分割图片
        input("\n按回车键开始分割图片...")
        train_counts, test_counts = self.split_images(train_per_class, test_per_class)

        # 3. 验证分割结果
        input("\n按回车键验证分割结果...")
        self.verify_split(train_per_class, test_per_class)

        # 4. 创建信息文件
        input("\n按回车键创建数据集信息文件...")
        self.create_dataset_info()

        # 5. 显示样本（可选）
        show_samples = input("\n是否显示样本图片？(y/n): ").lower()
        if show_samples == 'y':
            self.visualize_samples()

        print("\n" + "=" * 60)
        print("✅ 分割完成！")
        print(f"📁 训练集已保存到: {self.train_path}")
        print(f"📁 测试集已保存到: {self.test_path}")
        print("=" * 60)


# 使用示例
if __name__ == "__main__":
    # 设置你的MNIST数据集路径
    # 假设你的数据集结构是: mnist/train/0/, mnist/train/1/, ... mnist/train/9/
    SOURCE_PATH = "./mnist/train"  # 修改为你的实际路径

    # 创建分割器并运行
    splitter = MNISTSplitter(SOURCE_PATH)
    splitter.run(train_per_class=1500, test_per_class=500)