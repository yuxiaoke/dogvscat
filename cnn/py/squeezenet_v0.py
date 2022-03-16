#encoding:utf-8
import os

import torch.nn as nn
import torch.utils.data
import torchvision
from torch.autograd import Variable
from torchvision import transforms
import torchvision.models as models
import time
import logging
import warnings
import sys
# sys.exit(1)

warnings.filterwarnings("ignore")

num_epochs = 5
batch_size = 25
step_size = 1

#make modle
model = models.squeezenet1_0(pretrained=True)
# for parma in model.parameters():
#     parma.requires_grad = False
# model.classifier = nn.Sequential(nn.Dropout(p=0.5),
#                                  nn.Conv2d(512, 2, kernel_size=(1, 1), stride=(1, 1)),
#                                  nn.ReLU(inplace=True),
#                                  nn.AvgPool2d(kernel_size=13))
# for index, parma in enumerate(model.classifier.parameters()):
#     if index == 3:
#         parma.requires_grad = True
optimizer = torch.optim.Adam(model.classifier.parameters(), lr=0.001)
print("model", model)
# sys.exit(1)

#load data

data_transform = transforms.Compose([
    transforms.Scale((224,224), 2),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
         ])

train_dataset = torchvision.datasets.ImageFolder(root='/home/ubuntu/dukto/study/model/cat_vs_dog _cputest0417/data/train/',transform=data_transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=25, shuffle=True,num_workers=0)

val_dataset = torchvision.datasets.ImageFolder(root='/home/ubuntu/dukto/study/model/cat_vs_dog _cputest0417/data/val/', transform=data_transform)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=25, shuffle=True,num_workers=0)


#print(train_loader)
print("load dataset done")

model.train()
epoch = 0
#training data
# try:
for epoch in range(num_epochs):
    batch_size_start = time.time()
    running_loss = 0.0
    for i, (inputs,labels) in enumerate(train_loader):
        inputs = Variable(inputs)
        labels = Variable(labels)
        optimizer.zero_grad()
        outputs = model(inputs)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs, labels)        #交叉熵
        loss.backward()
        optimizer.step()      #更新权重
        running_loss += loss.data[0]
        if (i+1) % step_size ==0:
            print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f'
                  % (epoch, num_epochs, (i+1) / step_size, len(train_dataset) / (batch_size * step_size), running_loss / step_size))
            running_loss = 0.0

    print 'the %d num_epochs ' % (epoch+1),
    print 'need time %f' % (time.time() - batch_size_start)

    if (epoch+1) % 5 != 0:
        continue
    torch.save(model.state_dict(), './model/' + str(epoch) + "_resnet50_model.pkl")
    print('save the training model')
    correct = 0
    total = 0
    for j, (images,labels) in enumerate(val_loader):
        batch_size_start = time.time()
        images = Variable(images)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
        print("正确的数量：", correct)
        print(" Val BatchSize cost time :%.4f s" % (time.time() - batch_size_start))
        print('Test Accuracy of the model on the 5000 Val images: %.4f' % (float(correct) / total))
    if (float(correct) / total) >= 0.95:
        print('the Accuracy>=0.98 the num_epochs:%d'% epoch)
        break

print("training finish")

#save the training model25
torch.save(model.state_dict(), './model/resnet50_model.pkl')
print('save the training model')
# except:
#     torch.save(model.state_dict(), "./model/snopshot_" + str(epoch) + "_resnet50_model.pkl")
#     print('save snopshpot the training model Done.')


logging.basicConfig(level=logging.INFO,
                format ='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                datefmt ='%a, %d %b %Y %H:%M:%S',
                filename ='./txt/resnet50.txt',
                filemode ='w')

#model.load_state_dict(torch.load('model.pkl'))         #加载模型
