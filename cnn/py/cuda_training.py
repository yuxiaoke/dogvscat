#encoding:utf-8
import os

import torch.nn as nn
import torch.utils.data
import torchvision
import torch.nn.functional as F
#from tqdm import *
from torch.autograd import Variable
from torchvision import transforms
import time

num_epochs=6
#make modle
class alexnet(nn.Module):
    def __init__(self):
        super(alexnet, self).__init__()
        self.features = nn.Sequential(         # input shape (3, 224, 224)
            nn.Conv2d(
                in_channels=3,              # input height
                out_channels=96,            # n_filters
                kernel_size=11,              # filter size
                stride=4,                   # filter movement/step
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2),    #
            nn.Conv2d(96, 256, 5, 1, 2),
            nn.ReLU(),  # activation
            nn.MaxPool2d(kernel_size=3,stride=2),
            nn.Conv2d(256, 384, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(384, 384, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(384, 256, 3, 1, 1),
            nn.ReLU(),  # activation
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6,4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096,4096),
            nn.ReLU(),
            nn.Linear(4096, 2),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)           # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = self.classifier(x)
        return  x                                # return x for visualization


print("make modle done")

model = alexnet().cuda()
model.load_state_dict(torch.load("F:/work/pycharm/dogvscat/dogvscat/dogvscat/cnn/129_model.pkl"))

print(model)

#load data

data_transform = transforms.Compose([
    transforms.Scale((227,227),2),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
         ])

train_dataset = torchvision.datasets.ImageFolder(root='F:/work/pycharm/dogvscat/dogvscat/dogvscat/data/train/',transform=data_transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

val_dataset = torchvision.datasets.ImageFolder(root='F:/work/pycharm/dogvscat/dogvscat/dogvscat/data/val/', transform=data_transform)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=True)

cirterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#print(train_loader)
print("load dataset done")

model.train()
#training data
try:
    for epoch in range(num_epochs):
        batch_size_start = time.time()
        running_loss = 0.0
        for i, (inputs,labels) in enumerate(train_loader) :

            inputs = Variable(inputs).cuda()
            labels = Variable(labels).cuda()
            optimizer.zero_grad()
            outputs = model(inputs)
            criterion =nn.CrossEntropyLoss()
            loss = criterion(outputs,labels)        #交叉熵
            loss.backward()
            optimizer.step()      #更新权重
            running_loss +=loss.data[0]

            if (i+1) % 10 ==0:               #8 times
                print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f'
                      % (epoch + 1, num_epochs, (i+1) / 10, len(train_dataset) / (200 * 10),running_loss / 10))
                running_loss =0.0

        print('the %d num_epochs '% (epoch) ),
        print('need time %f' %(time.time() - batch_size_start))


        if (epoch+1) % 10 != 0 :
            continue
        torch.save(model.state_dict(), str(epoch) + "_model.pkl")
        print('save the training model')
        correct = 0
        total = 0
        for j, (images,labels) in enumerate(val_loader):
            batch_size_start = time.time()
            images = Variable(images).cuda()
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total +=labels.size(0)
            correct += (predicted == labels.cuda()).sum()
            print("正确的数量：", correct)
            print(" Val BatchSize cost time :%.4f s" % (time.time() - batch_size_start))
            print('Test Accuracy of the model on the 5000 Val images: %.4f' % (float(correct) / total))
        if (float(correct) / total)>=0.95:
            print('the Accuracy>=0.98 the num_epochs:%d'% epoch)
            break

    print("training finish")

    #save the training model25
    torch.save(model.state_dict(), 'model.pkl')
    print('save the training model')
except:
    torch.save(model.state_dict(), "snopshot_" + str(epoch) + "_model.pkl")
    print('save snopshpot the training model Done.')




#model.load_state_dict(torch.load('model.pkl'))         #加载模型
