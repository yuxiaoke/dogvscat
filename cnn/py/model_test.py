#encoding:utf-8

import os
import torch
import torch.nn as nn
import torch.utils.data
import torchvision
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms
import warnings
warnings.filterwarnings("ignore")

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
        return x                                # return x for visualization

model = alexnet()
model.load_state_dict = (torch.load("/home/ubuntu/dukto/study/model/cat_vs_dog _cputest0417/cnn/model/alexnet/alexnet-22_model.pkl",
                                    map_location=lambda storage, loc: storage))
print('model', model)

#test  dataset

data_transform = transforms.Compose([
    transforms.Scale((224, 224), 3),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[ 0.229, 0.224, 0.225])
    ])

test_dataset = torchvision.datasets.ImageFolder("/home/ubuntu/dukto/study/model/cat_vs_dog _cputest0417/data/test", data_transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=10, shuffle=False)
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