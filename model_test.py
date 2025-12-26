import torch
import torchvision.transforms as transforms
import torch.utils.data as data
from model import *
from torchvision.datasets import ImageFolder
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import glob
import os


class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir  # 图片根目录
        self.transform = transform  # 图像变换
        # 获取目录下所有图片路径（支持jpg、png、jpeg等格式）
        self.image_paths = glob.glob(os.path.join(root_dir, "*.[jp][pn]g"))  # 匹配jpg/png
        # 可选：排序保证图片顺序固定（便于后续对应结果）
        self.image_paths.sort()

    def __len__(self):
        # 返回数据集总样本数
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 加载单张图片
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")  # 先转RGB（兼容部分灰度图）

        # 应用图像变换
        if self.transform:
            image = self.transform(image)

        # 返回：图像张量 + 图片路径（便于后续关联预测结果）
        return image, img_path


def test_data_process():
    ROOT_TRAIN = '9x13_grid_mnist'
    test_transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),transforms.Resize((227,227)), transforms.ToTensor()])
    test_data = ImageDataset(root_dir=ROOT_TRAIN, transform=test_transform)


    test_loader = data.DataLoader(dataset=test_data,
                                   batch_size=1,
                                   shuffle=False,
                                   num_workers=0)
    return test_loader

def test_model(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    test_corrects = 0
    test_num = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            model.eval()
            output = model(images)
            pred = torch.argmax(output, dim=1)
            test_corrects += (pred == labels).float().sum().item()
            test_num += 1
        test_accuracy = test_corrects / test_num
        print('Test Accuracy: ', test_accuracy)


if __name__ == '__main__':
    model = AlexNet()
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    test_loader = test_data_process()
    # test_model(model, test_loader)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    with torch.no_grad():
        for imgs, paths in test_loader:
            imgs = imgs.to(device)
            output = model(imgs)
            pred = torch.argmax(output, dim=1)
            for img_path, pred in zip(paths, pred):
                print(f"图片 {img_path} 的预测结果：{pred.item()}")





