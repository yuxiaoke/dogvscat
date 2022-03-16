#encoding:utf-8
import os
import torch
import torch.nn as nn
import torch.utils.data
import torchvision
import torch.nn.functional as F
from tqdm import *
from torch.autograd import Variable
from torchvision import transforms

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
model = alexnet()
#test  dataset

data_transform = transforms.Compose([
    transforms.Scale((227, 227), 2),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[ 0.229, 0.224, 0.225])
    ])

test_dataset = torchvision.datasets.ImageFolder("f:/cat vs dog/data/test/", data_transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=50, shuffle=True)
cirterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#model = torch.load("model.pkl")
model = torch.load("/home/cm/dukto/cat vs dog/cnn/99_model.pkl")

model.eval()

correct=0
total =0
for  (images, labels) in enumerate(test_loader):
    images = Variable(images).cuda()
    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels.cuda*()).sum()
    print("正确的个数：",correct)
print('val accuracy of the 5000 val images:%4f' % (float(correct) / total))

