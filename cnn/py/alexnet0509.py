#encoding:utf-8
import os
import torch.nn as nn
import torch.utils.data
import torchvision
from torch.autograd import Variable
from torchvision import transforms
import time
import warnings
import torchvision.models as models
import sys
import matplotlib.pyplot as plt
import traceback


# 参数调整
# 实验的根目录
experimentDirRoot = "F:/work/pycharm/dogvscat/"
codeDirRoot = os.path.join(experimentDirRoot, "cnn")
dataDirRoot = os.path.join(experimentDirRoot, "data")
# 训练的Epoch
num_epochs = 5
batch_size = 16
# 多少个Epoch保存一个模型
saveModelEpoch = 8
# 多少个Epoch修改学习率
adjustLREpoch = 1
# 实验的后缀（第几次实验）
experimentSuffix = "_20180523_0"
# 多少个batchsize打印一次Loss
printLossBS = 100
# 初始化学习率
initLR = 0.001
# 学习变化率
factor = 0.2
# 多少batch_size后没有改善，学习率降低
patience = 125
#多少epoch减少一次学习率
step_size = 2

# def adjust_learning_rate(optimizer, lrRate):
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = param_group['lr'] * lrRate
#     print ("The LR is : ", param_group['lr'])

# 记录日志
class Logger(object):

    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

sys.stdout = Logger(os.path.join(codeDirRoot, "log", "alexnet%s.log"%experimentSuffix))

# 画折线图形并保存
def line_Loss_chart(x, y, picName):
    plt.figure(figsize=(20, 10))                         #创建绘图对象
    plt.plot(x, y, "b--", linewidth=1)   #在当前绘图对象绘图（X轴，Y轴，蓝色虚线，线宽度）

    plt.xlabel("batch_num")            #X轴标签
    plt.ylabel("Loss")               #Y轴标签
    plt.title("alexnet-Loss _Batch")          #图标题
    plt.savefig(picName)  # 保存图
# 画折线图形并保存
def line_chart(x_epoch, y_acc, picName):
    plt.figure()#创建绘图对象
    plt.plot(x_epoch, y_acc, "b--", linewidth=1)   #在当前绘图对象绘图（X轴，Y轴，蓝色虚线，线宽度）
    plt.ylim(0.00, 1.00)
    plt.xlabel("epoch")            #X轴标签
    plt.ylabel("accuracy")               #Y轴标签
    plt.title("alexnet-Line _chart")          #图标题
    # plt.savefig(os.path.join(codeDirRoot, "log", "pic", "resnet50%s.png"%experimentSuffix))  # 保存图
    plt.savefig(picName)  # 保存图

# 忽略警告
warnings.filterwarnings("ignore")
# 加载预训练模型
alexnet_model = models.alexnet(pretrained=False)
# for param in model.parameters():
#     param.requires_grad = False
alexnet_model.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 2),
        )
model = alexnet_model              
optimizer = torch.optim.SGD(model.parameters(), lr=initLR, momentum=1.0)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                               step_size=step_size, gamma=factor, last_epoch=-1)
print(model)


# load data
data_transform = transforms.Compose([
    transforms.Scale((224, 224), 2),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
         ])
train_dataset = torchvision.datasets.ImageFolder(root=os.path.join(dataDirRoot, "train"), transform=data_transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

val_dataset = torchvision.datasets.ImageFolder(root=os.path.join(dataDirRoot, "val"), transform=data_transform)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
print("load  dataset done")

# training data

x_epoch = []
y_acc = []
batchCountList = []
lossList = []
running_loss = 0.0
total=0
epoch = 0
try:
    for epoch in range(num_epochs):
        batch_size_start = time.time()
        model.train()
        for batch_count, (images, labels) in enumerate(train_loader):
            images = Variable(images)
            labels = Variable(labels)
            optimizer.zero_grad()
            outputs = model(images)
            criterion = nn.CrossEntropyLoss()
            loss = criterion(outputs, labels)        #交叉熵
            loss.backward()
            # optimizer.step()      #更新权重
            scheduler.step(loss.item())
            running_loss += loss.item()
            total += labels.size(0)
            if batch_count % 100   == 0:
                print('Epoch [%d/%d], Loss: %.8f'
                        %(epoch + 1, num_epochs, running_loss / total))

                batchCountList.append(batch_count + epoch * 20000 / batch_size)
                lossList.append(running_loss / total)

                picName = os.path.join(codeDirRoot, "log", "pic",
                                       "alexnet%s_LOSS.png"%experimentSuffix)

                line_Loss_chart(batchCountList, lossList, picName)

                running_loss = 0.0

        print('Epoch [%d/%d], need time %.8f'
              % (epoch + 1, num_epochs, time.time() - batch_size_start))

        correct = 0
        total = 0
        model.eval()
        for (images, labels) in val_loader:
            batch_size_start = time.time()
            images = Variable(images)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()

        # print("正确的数量：", correct)
        print(" Val BatchSize cost time :%.4f s" % (time.time() - batch_size_start))
        print('Test Accuracy of the model on the %d Val images: %.4f' % (total, float(correct) / total))
        if (float(correct) / total) >= 0.99:
            print('the Accuracy>=0.98 the num_epochs:%d'% epoch)
            break
        x_epoch.append(epoch)
        Acc = round((float(correct) / total), 3)
        y_acc.append(Acc)

        picName = os.path.join(codeDirRoot, "log", "pic",
                               "alexnet%s.png" % experimentSuffix)
        line_chart(x_epoch, y_acc, picName)

        # if (epoch + 1) % adjustLREpoch == 0:
        #     adjust_learning_rate(optimizer, LRModulus)

        if (epoch+1) % saveModelEpoch != 0:
            continue
        saveModelName = os.path.join(codeDirRoot, "model", "alexnet%s_model.pkl"%experimentSuffix + "_" + str(epoch))
        torch.save(model.state_dict(), saveModelName)



    #save the training model
    saveModelName = os.path.join(codeDirRoot, "model", "alexnet%s_model.pkl" % experimentSuffix + "_" + "Final")
    torch.save(model.state_dict(), saveModelName)
    print('save the training model')
except:
    traceback.print_exc()
    saveSnopModelName = os.path.join(codeDirRoot, "model",
                                     "alexnet%s_model.pkl" % experimentSuffix + "_" +
                                     "snopshot_" + str(epoch))
    torch.save(model.state_dict(), saveSnopModelName)
    print('save snopshpot the training model Done.')

