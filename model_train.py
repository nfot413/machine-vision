import copy
import time
from torchvision.datasets import ImageFolder
import torch
from torch import nn
import numpy as np
from torchvision import transforms
import torch.utils.data as data
import matplotlib.pyplot as plt
from model import AlexNet
import pandas as pd

def train_val_data_process():
    ROOT_TRAIN = 'mnist_train_1000_per_class'
    train_transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),transforms.Resize((227,227)), transforms.ToTensor()])
    train_data = ImageFolder(root=ROOT_TRAIN, transform=train_transform)

    train_data, val_data = data.random_split(train_data, [round(len(train_data)*0.8),round(len(train_data)-round(len(train_data)*0.8))])
    train_loader = data.DataLoader(dataset=train_data,
                                   batch_size=32,
                                   shuffle=True,
                                   num_workers=2)
    val_loader = data.DataLoader(dataset=val_data,
                                 batch_size=32,
                                 shuffle=True,
                                 num_workers=2)
    return train_loader, val_loader



def train_model(model, train_loader, val_loader, epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        print('Epoch {}/{}'.format(epoch + 1, epochs))
        print('-' * 10)
        train_loss = 0.0
        val_loss = 0.0
        train_acc = 0.0
        val_acc = 0.0
        train_num = 0
        val_num = 0
        for step, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            model.train()
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            train_acc += torch.sum(preds==labels.data)
            train_num += inputs.size(0)
        for step, (inputs, labels) in enumerate(val_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            model.eval()
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            val_acc += torch.sum(preds==labels.data)
            val_num += inputs.size(0)


        train_losses.append(train_loss/train_num)
        val_losses.append(val_loss/val_num)
        train_accs.append(train_acc.double().item()/train_num)
        val_accs.append(val_acc.double().item()/val_num)
        print('{} Train Loss: {:.4f} Train Acc: {:.4f} '.format(epoch+1,train_losses[-1],train_accs[-1]))
        print('{} Val Loss: {:.4f} Val Acc: {:.4f} '.format(epoch+1,val_losses[-1],val_accs[-1]))
        if val_accs[-1] > best_acc:
            best_acc = val_accs[-1]
            best_model_wts = copy.deepcopy(model.state_dict())
        time_use = time.time()-since
        print("时间{:.0f}m{:.0f}s".format(time_use//60, time_use%60))

    torch.save(best_model_wts,'./best_model.pth')
    train_process=pd.DataFrame(data={'epoch':range(1,epochs+1),
                                     'train_loss':train_losses,
                                     'train_acc':train_accs,
                                     'val_loss':val_losses,
                                     'val_acc':val_accs,})
    return train_process


def matplot_acc_loss(train_process):
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(train_process['epoch'],train_process['train_loss'],'ro-',label='train_loss')
    plt.plot(train_process['epoch'],train_process['val_loss'],'bs-',label='val_loss')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.subplot(1,2,2)
    plt.plot(train_process['epoch'],train_process['train_acc'],'ro-',label='train_acc')
    plt.plot(train_process['epoch'],train_process['val_acc'],'bs-',label='val_acc')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.show()



if __name__ == '__main__':
    AlexNet = AlexNet()
    train_loader, val_loader = train_val_data_process()
    train_process=train_model(AlexNet, train_loader, val_loader, epochs=20)
    matplot_acc_loss(train_process)
