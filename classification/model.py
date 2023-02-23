import torch
import torch.nn as nn

class Custom3DCNN(nn.Module):
    def __init__(self):
        super(Custom3DCNN, self).__init__()

        self.conv1 = nn.Conv3d(1, 64, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool3d(kernel_size=2)
        self.batchnorm1 = nn.BatchNorm3d(64)

        self.conv2 = nn.Conv3d(64, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool3d(kernel_size=2)
        self.batchnorm2 = nn.BatchNorm3d(64)

        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.maxpool3 = nn.MaxPool3d(kernel_size=2)
        self.batchnorm3 = nn.BatchNorm3d(128)

        self.conv4 = nn.Conv3d(128, 256, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU(inplace=True)
        self.maxpool4 = nn.MaxPool3d(kernel_size=2)
        self.batchnorm4 = nn.BatchNorm3d(256)

        self.globalavgpool = nn.AdaptiveAvgPool3d(1)

        self.fc1 = nn.Linear(256, 512)
        self.relu5 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(0.3)

        self.fc2 = nn.Linear(512, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x is of shape (batch_size, height, width, depth)
        # we add an extra dimension to represent the channel
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.batchnorm1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = self.batchnorm2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.maxpool3(x)
        x = self.batchnorm3(x)

        x = self.conv4(x)
        x = self.relu4(x)
        x = self.maxpool4(x)
        x = self.batchnorm4(x)

        x = self.globalavgpool(x)
        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.relu5(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.sigmoid(x)

        return x