#encoding:utf-8

import os
import torch
import torch.nn as nn
import torch.utils.data
import torchvision
import torch.nn.functional as F
# from tqdm import *
from torch.autograd import Variable
from torchvision import transforms
import torchvision.models as models
import warnings
import logging
import sys

warnings.filterwarnings("ignore")

class Logger(object):

    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass
sys.stdout = Logger("./txt/vgg16-1test.txt")

model = models.vgg16(pretrained=False)
model.classifier = nn.Sequential(nn.Linear(25088, 4096),
                                 nn.ReLU(),
                                 nn.Dropout(p=0.5),
                                 nn.Linear(4096, 4096),
                                 nn.ReLU(),
                                 nn.Dropout(p=0.5),
                                 nn.Linear(4096, 2))
print("model", model)
model.load_state_dict(torch.load(
    "/home/ubuntu/dukto/study/model/cat_vs_dog _cputest0417/cnn/model/vgg16/39_vgg16_model.pkl",
                                 map_location=lambda storage, loc: storage))
print("model done")
# model.load_state_dict(torch.load('model.pkl'))

#test  dataset
data_transform = transforms.Compose([
    transforms.Scale((224, 224), 2),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
test_dataset = torchvision.datasets.ImageFolder("/home/ubuntu/dukto/study/model/cat_vs_dog _cputest0417/data/test", data_transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True)
print("load dataset done")
model.eval()
correct = 0
total = 0
for images, labels in test_loader:
    images = Variable(images)
    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()
print("正确的数量%d,所有图片数量%d：" % (correct, total))
print('val accuracy of the %d val images:%.4f' % (total, float(correct) / total))




#model.load_state_dict(torch.load('model.pkl'))