#encoding:utf-8

import os
import torch
import torch.nn as nn
import torch.utils.data
import torchvision
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt
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
sys.stdout = Logger("./txt/show_image.txt")

#显示图片函数
def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)
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

model = alexnet()

# model = models.resnet50(pretrained=False)
# model.fc = nn.Linear(2048, 2)
print('model', model)
model.load_state_dict(torch.load("/home/ubuntu/dukto/study/model/cat_vs_dog _cputest0417/cnn/model/alexnet/alexnet-22_model.pkl", map_location=lambda storage, loc: storage))
# model.load_state_dict(torch.load('model.pkl'))

#test  dataset
data_transform = transforms.Compose([
    transforms.Scale((224, 224), 2),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

#网上的图片
test_dataset = torchvision.datasets.ImageFolder("/home/ubuntu/dukto/study/model/cat_vs_dog _cputest0417/data/show", data_transform)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True)

class_names = test_dataset.classes

# 显示一些图片预测函数
def visualize_model(model, num_images):
    model.eval()
    images_so_far = 0

    for i, data in enumerate(test_loader):
        inputs, labels = data
        inputs, labels = Variable(inputs), Variable(labels)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)

        for j in range(inputs.size()[0]):
            images_so_far += 1
            ax = plt.subplot(num_images//2, 2, images_so_far)
            ax.axis('off')
            ax.set_title('predicted: {}'.format(class_names[predicted[j]]))
            imshow(inputs.cpu().data[j])
            if images_so_far == num_images:
                return
visualize_model(model, 2)

plt.ioff()
plt.show()


#model.load_state_dict(torch.load('model.pkl'))