import torch
import torchvision.transforms as transforms
import torch.utils.data as data
from model import *
from torchvision.datasets import FashionMNIST

# def test_data_process():
#     test_data = FashionMNIST(root='./data',
#                               train=False,
#                               download=True,
#                               transform=transforms.Compose([transforms.Resize(size=227), transforms.ToTensor()]))
#     test_loader = data.DataLoader(dataset=test_data,
#                                    batch_size=1,
#                                    shuffle=True,
#                                    num_workers=0)
#     return test_loader
#
# def test_model(model, test_loader):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = model.to(device)
#     test_corrects = 0
#     test_num = 0
#     with torch.no_grad():
#         for images, labels in test_loader:
#             images, labels = images.to(device), labels.to(device)
#             model.eval()
#             output = model(images)
#             pred = torch.argmax(output, dim=1)
#             test_corrects += (pred == labels).float().sum().item()
#             test_num += 1
#         test_accuracy = test_corrects / test_num
#         print('Test Accuracy: ', test_accuracy)
#
#
# if __name__ == '__main__':
#     model = AlexNet()
#     model.load_state_dict(torch.load('best_model.pth'))
#     test_loader = test_data_process()
#     test_model(model, test_loader)
#     # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     # model = model.to(device)
#     # with torch.no_grad():
#     #     for images, labels in test_loader:
#     #         images, labels = images.to(device), labels.to(device)
#     #         model.eval()
#     #         output = model(images)
#     #         pred = torch.argmax(output, dim=1)
#     #         result = pred.item()
#     #         label = labels.item()
#     #         print('预测值:',result,'--------','真实值',label)
#     #
#
#


