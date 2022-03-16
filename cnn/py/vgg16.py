#encoding:utf-8
import os
import torch.nn as nn
import torch.utils.data
import torchvision
from torch.autograd import Variable
from torchvision import transforms
import torchvision.models as models
import time
from pylab import *                                 #支持中文
import warnings
import sys
import matplotlib.pyplot as plt
import traceback
import logging

warnings.filterwarnings("ignore")
lr1 = 0.001
lr2 = 0.0001
class Logger(object):

    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass
sys.stdout = Logger("./txt/vgg16_4.txt")

#打印图形函数并保存图形
def line_chart(x_epoch, y_acc):
    plt.figure()                         #创建绘图对象
    plt.plot(x_epoch, y_acc, "b--", linewidth=1)   #在当前绘图对象绘图（X轴，Y轴，蓝色虚线，线宽度）
    plt.ylim(0.00, 1.00)
    plt.xlabel("epoch")            #X轴标签
    plt.ylabel("accuracy")               #Y轴标签
    plt.title("vgg16-Line _chart")          #图标题
    plt.show()                           #显示图
    plt.savefig("./txt/vgg16.png")       #保存图


num_epochs = 100
batch_size = 32

#make modle
model = models.vgg16(pretrained=False)
# for parma in model.parameters():
#     parma.requires_grad = False
model.classifier = nn.Sequential(nn.Linear(25088, 4096),
                                 nn.ReLU(),
                                 nn.Dropout(p=0.5),
                                 nn.Linear(4096, 4096),
                                 nn.ReLU(),
                                 nn.Dropout(p=0.5),
                                 nn.Linear(4096, 2))
# for index, parma in enumerate(model.classifier.parameters()):
#     if index == 6:
#         parma.requires_grad = True

# optimizer = torch.optim.SGD(model.classifier.parameters(), lr=lr1)
print("model", model)
# sys.exit(0)


#load data
data_transform = transforms.Compose([
    transforms.Scale((224,224), 2),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
         ])
train_dataset = torchvision.datasets.ImageFolder(root='/home/ubuntu/dukto/study/model/cat_vs_dog _cputest0417/data/train/',transform=data_transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle=True, num_workers=0)

val_dataset = torchvision.datasets.ImageFolder(root='/home/ubuntu/dukto/study/model/cat_vs_dog _cputest0417/data/val/', transform=data_transform)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = batch_size, shuffle=True, num_workers=0)

#print(train_loader)
print("load dataset done")

model.train()
epoch = 0
x_epoch = []
y_acc = []
#training data
try:
    for epoch in range(num_epochs):
        batch_size_start = time.time()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            if epoch >= 5:
                optimizer = torch.optim.SGD(model.classifier.parameters(), lr=lr2)
                print("lr", lr2)
            else:
                optimizer = torch.optim.SGD(model.classifier.parameters(), lr=lr1)
                print("lr", lr1)
            inputs = Variable(inputs)
            labels = Variable(labels)
            optimizer.zero_grad()
            outputs = model(inputs)
            criterion = nn.CrossEntropyLoss()
            loss = criterion(outputs, labels)        #交叉熵
            loss.backward()
            optimizer.step()      #更新权重
            running_loss += loss.data[0]

        print('Epoch [%d/%d], Loss: %.4f,need time %.4f'
                  % (epoch + 1, num_epochs, running_loss / (4000 / batch_size), time.time() - batch_size_start))


        # print 'the %d num_epochs '% (epoch) ,
        # print 'need time %f' %(time.time() - batch_size_start)


        # if (epoch+1) % 10 != 0 :
        #     continue
        torch.save(model.state_dict(), './model/VGG16_4' + str(epoch) + "_model.pkl")
        print('save the training model')
        correct = 0
        total = 0
        for j, (images, labels) in enumerate(val_loader):
            batch_size_start = time.time()
            images = Variable(images)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total +=labels.size(0)
            correct += (predicted == labels).sum()

        print(" Val BatchSize cost time :%.4f s" % (time.time() - batch_size_start))
        print('Test Accuracy of the model on the 5000 Val images: %.4f' % (float(correct) / total))
        if (float(correct) / total)>=0.95:
            print('the Accuracy>=0.98 the num_epochs:%d'% epoch)
            break
        x_epoch.append(epoch)
        Acc = round((float(correct) / total), 3)
        y_acc.append(Acc)
        line_chart(x_epoch, y_acc)
    print("training finish")

    #save the training model25
    torch.save(model.state_dict(), './model/VGG16_4_model.pkl')
    print('save the training model')
except:
    torch.save(model.state_dict(), "./model/snopshot_VGG16_4" + str(epoch) + "_model.pkl")
    print('save snopshpot the training model Done.')



#model.load_state_dict(torch.load('model.pkl'))         #加载模型
