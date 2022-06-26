import json
import random

import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1 import host_subplot
from numpy import mean
from paddle.io import DataLoader, Dataset
from PIL import Image

import matplotlib.pyplot as plt
import paddle
import paddle.nn.functional as F
import paddle.vision.transforms as T



import warnings

from sklearn.metrics import f1_score

warnings.filterwarnings("ignore")


label_json = pd.read_json('label.json')
label_json['filename'] = label_json['annotations'].apply(lambda x: x['filename'].replace('\\', '/'))
label_json['period'] = label_json['annotations'].apply(lambda x: x['period'])
label_json['weather'] = label_json['annotations'].apply(lambda x: x['weather'])


print(label_json.head())
print(label_json["weather"].value_counts())

label_json['period'], period_dict = pd.factorize(label_json['period'])
label_json['weather'], weather_dict = pd.factorize(label_json['weather'])
print(label_json['period'].value_counts())

print(label_json['weather'].value_counts())

#print(label_json)
# 自定义数据集
class WeatherDataset(Dataset):
    def __init__(self, df):
        super(WeatherDataset, self).__init__()
        self.df = df

        # 定义数据扩增方法
        self.transform = T.Compose([
            T.Resize(size=(340, 340)),
            T.RandomCrop(size=(256, 256)),
            T.RandomRotation(10),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.ToTensor(),
            T.Normalize(mean=0.5, std=0.5)
        ])

    def __getitem__(self, index):
        file_name = self.df['filename'].iloc[index]
        img = Image.open(file_name)
        img = self.transform(img)
        return img, \
               paddle.to_tensor(self.df['period'].iloc[index]), \
               paddle.to_tensor(self.df['weather'].iloc[index])

    def __len__(self):
        return len(self.df)

# 训练集
train_dataset = WeatherDataset(label_json.iloc[:-600])
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

print(train_dataset)
print(type(train_dataset))

# 验证集
val_dataset = WeatherDataset(label_json.iloc[-600:])
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)

from paddle.vision.models import resnet18

# 自定义模型，模型有两个head
class WeatherModel(paddle.nn.Layer):
    def __init__(self):
        super(WeatherModel, self).__init__()
        backbone = resnet18(pretrained=True)
        backbone.fc = paddle.nn.Identity()
        self.backbone = backbone

        # 分类1
        self.fc1 = paddle.nn.Linear(512, 4)

        # 分类2
        self.fc2 = paddle.nn.Linear(512, 3)

    def forward(self, x):
        out = self.backbone(x)

        # 同时完成类别1 和 类别2 分类
        logits1 = self.fc1(out)
        logits2 = self.fc2(out)
        return logits1, logits2

def plot_acc_loss(loss, acc, flag, trainortest, epoch):
    plt.figure()
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




model = WeatherModel()
model(paddle.to_tensor(np.random.rand(10, 3, 256, 256).astype(np.float32)))

# 定义损失函数和优化器
optimizer = paddle.optimizer.Adam(parameters=model.parameters(), learning_rate=0.0001)
criterion = paddle.nn.CrossEntropyLoss()


'''
print(len(train_loader))
print(len(val_loader))
'''
train_all_loss, train_all_acc1, train_all_acc2, train_all_Sc = [], [], [], []
test_all_loss, test_all_acc1, test_all_acc2, test_all_Sc = [], [], [], []
test = []
for epoch in range(0, 1000):

    print("\n\nEPOCH:", epoch)

    Train_Loss, Val_Loss = [], []
    Train_ACC1, Train_ACC2 = [], []
    Val_ACC1, Val_ACC2 = [], []

    # 模型训练
    model.train()
    Fscore_w_train, Fscore_p_train, Fscore_all_train = [], [], []  # 训练过程中每一轮次的Fscore记录
    acc_w_train, acc_p_train = [], []  # 训练过程中每一轮次的acc记录
    loss_list = []  # 训练过程中每一轮次的loss记录
    for i, (x, y1, y2) in enumerate(train_loader):
        pred1, pred2 = model(x)

        # 类别1 loss + 类别2 loss 为总共的loss
        loss = criterion(pred1, y1) + criterion(pred2, y2)
        Train_Loss.append(loss.item())
        loss.backward()
        optimizer.step()
        optimizer.clear_grad()

        Train_ACC1.append((pred1.argmax(1) == y1.flatten()).numpy().mean())
        Train_ACC2.append((pred2.argmax(1) == y2.flatten()).numpy().mean())

        # f1_score
        f1_score1 = f1_score(y1.flatten().cpu(), pred1.argmax(1).cpu(), average='macro')
        f1_score2 = f1_score(y2.flatten().cpu(), pred2.argmax(1).cpu(), average='macro')
        f1_all = (f1_score2 + f1_score1) / 2.0
        # 记录f1_score
        Fscore_w_train.append(f1_score1)
        Fscore_p_train.append(f1_score2)
        Fscore_all_train.append(f1_all)  # 单个训练过程中Fscore记录

    print("\nTrain数据记录中:\nLoss:")

    print("训练中loss记录:", mean(Train_Loss))

    print("F-score:")
    print("Weather类别---F-score得分：", mean(Fscore_w_train))
    print("Period类别---F-score得分：", mean(Fscore_p_train))
    print("F-score平均总得分：", mean(Fscore_all_train))

    print("ACC:")
    # print("Weather类别---acc记录：", acc_w_train)
    # print("Period类别---acc记录：", acc_p_train)
    print("Weather类别---训练准确率acc: %f%%" % (mean(Train_ACC1) * 100))
    print("Period类别---训练准确率acc: %f%%" % (mean(Train_ACC2) * 100))

    train_all_loss.extend(Train_Loss)
    train_all_acc1.extend(Train_ACC1)
    train_all_acc2.extend(Train_ACC2)
    train_all_Sc.extend(Fscore_all_train)








    # 模型验证
    model.eval()

    Fscore_w_test, Fscore_p_test, Fscore_f1score = [], [], []


    for i, (x, y1, y2) in enumerate(val_loader):
        pred1, pred2 = model(x)
        loss = criterion(pred1, y1) + criterion(pred2, y2)
        Val_Loss.append(loss.item())

        f1_score1 = f1_score(y1.flatten().cpu(), pred1.argmax(1).cpu(), average='macro')
        f1_score2 = f1_score(y2.flatten().cpu(), pred2.argmax(1).cpu(), average='macro')
        f1_all = (f1_score2 + f1_score1) / 2.0
        Fscore_w_test.append(f1_score1)
        Fscore_p_test.append(f1_score2)
        Fscore_f1score.append(f1_all)

        Val_ACC1.append((pred1.argmax(1) == y1.flatten()).numpy().mean())
        Val_ACC2.append((pred2.argmax(1) == y2.flatten()).numpy().mean())

    print("\nTEST数据记录中:\nLoss:\n测试中loss记录:", mean(Val_Loss))
    print("Fscore:")
    print("Weather类别的F-score得分：", mean(Fscore_w_test))
    print("Period类别的F-score得分：", mean(Fscore_p_test))
    print("总F-score得分：", mean(Fscore_f1score))
    print("ACC:")
    print("weather类别的ACC:%f%%" % (mean(Val_ACC1) * 100))
    print("period类别的ACC:%f%%" % (mean(Val_ACC2) * 100))

    test_all_loss.extend(Val_Loss)
    test_all_acc1.extend(Val_ACC1)
    test_all_acc2.extend(Val_ACC2)
    test_all_Sc.extend(Fscore_f1score)

    if epoch % 5 == 0:  # and epoch != animals
        print("**************************************************************")
        print("\n\n又过了10个epoch啦!")
        print("Train:")
        print("本轮训练中Loss---", mean(Train_Loss))
        print("本轮训练中ACC1---", mean(Train_ACC1))
        print("本轮训练中ACC2---", mean(Train_ACC2))
        print("本轮训练中Fscore---", mean(Fscore_all_train))

        print("Test:")
        print("本轮测试中Loss---", mean(Val_Loss))
        print("本轮测试中ACC1---", mean(Val_ACC1))
        print("本轮测试中ACC2---", mean(Val_ACC2))
        print("本轮测试中Fscore---", mean(Fscore_f1score))

        print("\n\n**************************************************************")

        plot_acc_loss(train_all_loss, test, 1, 1, epoch)
        plot_acc_loss(test, train_all_acc1, 1, 1, epoch)
        plot_acc_loss(test, train_all_acc2, 1, 1, epoch)
        plot_acc_loss(test, train_all_Sc, 0, 1, epoch)

        plot_acc_loss(test_all_loss, test, 1, 0, epoch)
        plot_acc_loss(test, test_all_acc1, 1, 0, epoch)
        plot_acc_loss(test, test_all_acc2, 1, 0, epoch)
        plot_acc_loss(test, test_all_Sc, 0, 0, epoch)

        print("\n\nComplete!")