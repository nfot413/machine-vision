# 浙江大学 2025-2026 秋冬 机器视觉与图像处理大作业

## 项目简介
本项目基于 PyTorch 实现了一个轻量级 CNN，用于手写数字识别（0-9）。项目包含训练/验证流程、命令行推理与测试评估，并提供 Streamlit Web 界面，支持单张数字图片识别以及 9x13 方格纸照片的自动拉平、分割与批量识别。

## 文件构成
- `app.py`：Streamlit Web 应用入口，包含训练、单张推理、测试集评估、方格纸批量识别功能。
- `src/models/CNN.py`：CNN 模型结构与权重初始化。
- `src/train/train.py`：训练脚本，按类别采样数据并保存模型到 `checkpoints/cnn.pt`。
- `src/infer/infer.py`：命令行单张图片推理与模型加载工具。
- `src/infer/infer_test.py`：测试集整体准确率与每类准确率评估。
- `src/preprocess/flatten_grid.py`：方格纸照片透视拉平预处理。
- `src/preprocess/split_grid.py`：去网格线、切分 9x13 网格并生成 28x28 子图。
- `src/utils/utils.py`：设备选择与图像预处理（灰度化、缩放、Tensor 化）。
- `data/train/`：训练集目录（`ImageFolder` 格式，按类别文件夹 0-9）。
- `data/test/`：测试集目录（与训练集相同结构）。
- `checkpoints/`：训练产生的模型权重存放目录。
- `requirements.txt`：依赖列表。

## 使用方法
### 1) 安装依赖
```bash
pip install -r requirements.txt
```

### 2) 训练模型
```bash
python src/train/train.py
```
训练完成后会保存权重到 `checkpoints/cnn.pt`。

### 3) 命令行单张推理
```bash
python src/infer/infer.py --image path/to/your.png --ckpt checkpoints/cnn.pt
```

### 4) 测试集评估
```bash
python src/infer/infer_test.py --ckpt checkpoints/cnn.pt
```
输出总体准确率与每类准确率。

### 5) 启动 Web 界面
```bash
streamlit run app.py
```
在浏览器中可进行训练、单张推理、测试集评估以及 9x13 方格纸批量识别。

### 6) 方格纸预处理脚本（可选）
```bash
# 透视拉平
python src/preprocess/flatten_grid.py --image path/to/photo.jpg --out path/to/flat.png

# 切分为 117 张 28x28 子图
python src/preprocess/split_grid.py --image path/to/flat.png --out_dir path/to/out_dir
```
