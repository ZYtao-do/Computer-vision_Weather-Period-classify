import glob
import json
import os
import random
import re

import numpy as np
import pandas
from PIL import Image
import torch.nn as nn
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import host_subplot
from numpy import mean
from sklearn.metrics import f1_score
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch
from torchvision.models import resnet18

'''
[animals] 时间:  ['Morning', 'Afternoon', 'Dusk', 'Dawn']  animals animals food scenery      ['AM', 'PM']        animals animals
[food] 天气:  ['Cloudy', 'Sunny', 'Rainy']   animals animals food                   ['Good', 'Bad']     animals animals
'''


def read_json(path):
    """
    处理json数据，提取出其中的file_name, period, weather存入字典中
    """
    with open(path, 'r') as f:
        json_data = json.loads(f.read())
    dataset_all = pandas.json_normalize(json_data, record_path=['annotations'])  # 展开内嵌的 JSON 数据 annotations。
    print(dataset_all)
    # print(dataset_all['filename'])
    print(dataset_all.head())
    print("\n天气数据统计：")
    print(dataset_all["weather"].value_counts())
    print("\n时间数据统计：")
    print(dataset_all["period"].value_counts())
    print("\n路径数据统计：")
    print(dataset_all["filename"].value_counts())

    # 前者 dataset 为 DataFrame 类别； 类别string -> 编号
    # 后者 period_dic 为 index 类别，可直接索引
    dataset_all['weather'], weather_dic = pandas.factorize(dataset_all['weather'])
    dataset_all['period'], period_dic = pandas.factorize(dataset_all['period'])
    print(dataset_all["weather"].value_counts())
    print(dataset_all["period"].value_counts())

    keys = list(dataset_all.keys())
    values = list(dataset_all.values)

    # print(keys[animals])
    # print(values[animals])
    # print(values[animals][food])

    print(weather_dic)
    print(period_dic)
    # print(dataset_all['weather'].iloc[animals])  # 获取 dataset_all 中样例2的天气类别
    return dataset_all, values


def files(path):  # 遍历path下所有文件
    file = os.listdir(path)  # files是一个列表
    # print(file)
    files_path = []
    for i in range(len(file)):
        a = path + "/" + file[i]
        files_path.append(a)
    return files_path, file


class DataSetTT(torch.utils.data.Dataset):
    def __init__(self, img_dataset, img_tranform, img_path, train_flag):
        super(DataSetTT, self).__init__()
        self.img_Dataset = img_dataset
        self.img_path = img_path.copy()
        self.img_tranform = img_tranform

        weather_lb = []
        period_lb = []
        for i in range(len(self.img_path)):
            if os.path.isfile(self.img_path[i]):
                if self.img_Dataset['weather'].iloc[i] == 1:  # Good
                    self.img_Dataset['weather'].iloc[i] = 1
                elif self.img_Dataset['weather'].iloc[i] == 0 or self.img_Dataset['weather'].iloc[i] == 2:  # Bad
                    self.img_Dataset['weather'].iloc[i] = 0
                if self.img_Dataset['period'].iloc[i] == 0 or self.img_Dataset['period'].iloc[i] == 3:  # AM
                    self.img_Dataset['period'].iloc[i] = 1
                elif self.img_Dataset['period'].iloc[i] == 1 or self.img_Dataset['period'].iloc[i] == 2:  # PM
                    self.img_Dataset['period'].iloc[i] = 0
                if train_flag == 1:
                    weather_lb.append(int(self.img_Dataset['weather'].iloc[i]))
                    period_lb.append(int(self.img_Dataset['period'].iloc[i]))
                elif train_flag == 0:
                    weather_lb.append(int(self.img_Dataset['weather'].iloc[i + 1733]))
                    period_lb.append(int(self.img_Dataset['period'].iloc[i + 1733]))
        # print(images)
        self.weather_lb = weather_lb
        self.period_lb = period_lb

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, index):
        img_path = self.img_path[index]
        weather_lb = np.array(int(self.weather_lb[index]))
        period_lb = np.array(int(self.period_lb[index]))
        weather_lb = torch.from_numpy(weather_lb).long()
        period_lb = torch.from_numpy(period_lb).long()

        img_all = Image.open(img_path)

        if self.img_tranform:
            img_all = self.img_tranform(img_all)
        # print(type(img_all))
        # print(type(weather_lb))
        return img_all, period_lb, weather_lb


class ClassifyWpModel(nn.Module):
    def __init__(self):
        super(ClassifyWpModel, self).__init__()
        model = resnet18(pretrained=True)  # 加载已经训练好的模型

        model.fc = torch.nn.Identity()

        self.backbone = model

        self.weather_lb = nn.Linear(in_features=512, out_features=3, bias=True)
        self.period_lb = nn.Linear(in_features=512, out_features=4, bias=True)

    def forward(self, x):
        out = self.backbone(x)

        classify1 = self.period_lb(out)
        classify2 = self.weather_lb(out)

        return classify1, classify2


def plot_acc_loss(loss, acc, flag, trainortest, epoch):
    host = host_subplot(111)
    plt.subplots_adjust(right=0.8)  # ajust the right boundary of the plot window
    par1 = host.twinx()  # 共享x轴

    # 设置轴信息
    host.set_xlabel("steps")
    host.set_ylabel("loss")
    # par1.set_ylabel("evaluate")

    # 配置相关参数
    p1, = host.plot(range(len(loss)), loss, label="loss")
    if flag == 1:
        par1.set_ylabel("accuracy")
        p2, = par1.plot(range(len(acc)), acc, label="accuracy")
    else:
        par1.set_ylabel("Fscore")
        p2, = par1.plot(range(len(acc)), acc, label="Fscore")

    host.legend(loc=5)

    # set label color
    host.axis["left"].label.set_color(p1.get_color())
    par1.axis["right"].label.set_color(p2.get_color())

    plt.draw()
    name = random.randint(0, 100000)
    if trainortest == 1:
        plt.savefig('./train_pic/hello_%d_%d.jpg' % (epoch, name))
    else:
        plt.savefig('./test_pic/hello_%d_%d.jpg' % (epoch, name))
    # plt.show()
    plt.close()


def evalute(model, test_loader, device, criterion):
    model.eval()
    test_list = []
    Test_loss = []
    Fscore_w_test, Fscore_p_test, Fscore_f1score = [], [], []
    acc_w_test, acc_p_test = [], []
    # print(len(test_loader))
    for i, (images, w_label, p_label) in enumerate(test_loader):
        images, w_label, p_label = images.to(device), w_label.to(device), p_label.to(device)
        pred1, pred2 = model(images)

        # loss
        # 类别1 loss1 + 类别2 loss2  = 总共的loss
        loss1 = criterion(pred1, w_label)
        loss2 = criterion(pred2, p_label)
        loss = (loss1 + loss2) / 2.0
        Test_loss.append(loss.item())
        # acc
        acc_w_test.append((pred1.argmax(1) == w_label.flatten()).cpu().numpy().mean())
        acc_p_test.append((pred2.argmax(1) == p_label.flatten()).cpu().numpy().mean())
        # F-score
        f1_score1 = f1_score(w_label.flatten().cpu(), pred1.argmax(1).cpu(), average='macro')
        f1_score2 = f1_score(p_label.flatten().cpu(), pred2.argmax(1).cpu(), average='macro')
        f1_all = (f1_score2 + f1_score1) / 2.0

        Fscore_w_test.append(f1_score1)
        Fscore_p_test.append(f1_score2)
        Fscore_f1score.append(f1_all)

    print("\nTEST数据记录中:\n总Loss：", mean(Test_loss))
    print("Weather类别的F-score得分：", mean(Fscore_w_test))
    print("Period类别的F-score得分：", mean(Fscore_p_test))
    print("weather类别的ACC:%d%%" % (mean(acc_w_test) * 100))
    print("period类别的ACC:%d%%" % (mean(acc_p_test) * 100))

    return Test_loss, acc_w_test, acc_p_test, Fscore_f1score


def train(model, train_loader, device, test_loader, optimizer, criterion):

    model = model.to(device)
    # 定义损失函数和优化器

    # 模型训练
    model.train()

    Fscore_w_train, Fscore_p_train, Fscore_all_train = [], [], []  # 训练过程中每一轮次的Fscore记录
    acc_w_train, acc_p_train = [], []  # 训练过程中每一轮次的acc记录
    loss_list = []  # 训练过程中每一轮次的loss记录

    for i, (images, w_label, p_label) in enumerate(train_loader):
        images, w_label, p_label = images.to(device), w_label.to(device), p_label.to(device)
        label1, label2 = model(images)

        # 类别1 loss1 + 类别2 loss2  = 总共的loss
        loss1 = criterion(label1, w_label)
        loss2 = criterion(label2, p_label)
        loss = (loss1 + loss2) / 2.0

        loss_list.append(loss.item())  # 单个训练过程中Loss记录

        loss.backward()  # 反向传播
        optimizer.step()
        optimizer.zero_grad()  # #梯度置零，是把loss关于weight的导数变成0.

        # acc
        acc_w_train.append((label1.argmax(1) == w_label.flatten()).cpu().numpy().mean())
        acc_p_train.append((label2.argmax(1) == p_label.flatten()).cpu().numpy().mean())  # 单个训练过程中ACC记录
        # f1_score
        f1_score1 = f1_score(w_label.flatten().cpu(), label1.argmax(1).cpu(), average='macro')
        f1_score2 = f1_score(p_label.flatten().cpu(), label2.argmax(1).cpu(), average='macro')
        f1_all = (f1_score2 + f1_score1) / 2.0
        # 记录f1_score
        Fscore_w_train.append(f1_score1)
        Fscore_p_train.append(f1_score2)
        Fscore_all_train.append(f1_all)  # 单个训练过程中Fscore记录

        print("\nTrain数据记录中:\nLoss:")

        print("训练中loss记录:", mean(loss_list))

        print("F-score:")
        print("Weather类别---F-score得分：", mean(Fscore_w_train))
        print("Period类别---F-score得分：", mean(Fscore_p_train))
        print("F-score平均总得分：", mean(Fscore_all_train))

        print("ACC:")
        # print("Weather类别---acc记录：", acc_w_train)
        # print("Period类别---acc记录：", acc_p_train)
        print("Weather类别---训练准确率acc: %d%%" % (mean(acc_w_train) * 100))
        print("Period类别---训练准确率acc: %d%%" % (mean(acc_p_train) * 100))

        if epoch % 50 == 0 and epoch != 0:
            print("\n\n距离上一次已经训练了50个epoch！")

            print("保存模型中...")
            torch.save(model, './model/model_%d.pwf' % epoch)
        return loss_list, acc_w_train, acc_p_train, Fscore_all_train


if __name__ == '__main__':
    json_path = "./label.json"
    dataset_all, dataset_list = read_json(json_path)
    print(type(dataset_list))

    # 设置训练集、测试集路径
    train_img_path, train_xh = files(
        './train_images')
    test_img_path, test_xh = files(
        './test_images')
    print(train_img_path)

    # 设置 GPU or CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 转换图片格式
    img_tranform = transforms.Compose([
        transforms.Resize((340, 340)),
        transforms.RandomCrop((256, 256)),
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=False)
    ])

    train_Dataset = DataSetTT(img_dataset=dataset_all, img_tranform=img_tranform, img_path=train_img_path, train_flag=1)
    test_Dataset = DataSetTT(img_dataset=dataset_all, img_tranform=img_tranform, img_path=test_img_path, train_flag=0)
    print(type(train_Dataset))

    # 训练集数据
    train_loader = DataLoader(train_Dataset, batch_size=64, shuffle=True)
    # print(train_loader)

    # 测试集数据
    test_loader = DataLoader(test_Dataset, batch_size=64, shuffle=True)

    # 加载模型
    model = ClassifyWpModel()
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = torch.nn.CrossEntropyLoss()

    train_all_loss, train_all_acc, train_all_Sc = [], [], []
    test = []
    for epoch in range(0, 1000):
        print("\n\nEPOCH:", epoch)
        train_loss, train_acc1, train_acc2, train_Fscore = train(model, train_loader, device, test_loader, optimizer, criterion)
        train_all_loss.extend(train_loss)
        train_all_acc.extend(train_acc1)
        train_all_Sc.extend(train_Fscore)

        test_loss, test_acc1, test_acc2, test_Fscore = evalute(model, test_loader, device, criterion)

        if epoch % 10 == 0:  # and epoch != animals
            print("**************************************************************")
            print("\n\n又过了10个epoch啦!")
            print("Train:")
            print("本轮训练中Loss---", mean(train_loss))
            print("本轮训练中ACC1---", mean(train_acc1))
            print("本轮训练中ACC2---", mean(train_acc2))
            print("本轮训练中Fscore---", mean(train_Fscore))

            print("Test:")
            print("本轮测试中Loss---", mean(test_loss))
            print("本轮测试中ACC1---", mean(test_acc1))
            print("本轮测试中ACC2---", mean(test_acc2))
            print("本轮测试中Fscore---", mean(test_Fscore))


            print("\n\n**************************************************************")

            plot_acc_loss(train_all_loss, test, 1, 0, epoch)
            plot_acc_loss(test, train_all_acc, 1, 0, epoch)
            plot_acc_loss(test, train_all_Sc, 0, 0, epoch)

    print("\n\nComplete!")
