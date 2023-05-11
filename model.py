import torch
import torch.nn as nn


class UNet3D(nn.Module):

    def __init__(self, n_class=1):
        super(UNet3D, self).__init__()

        self.conv1a = nn.Conv3d(1, 64, kernel_size=3, padding=1)
        self.bn1a = nn.BatchNorm3d(64)
        self.activation1a = nn.ReLU(64)
        self.conv1b = nn.Conv3d(64, 64, kernel_size=3, padding=1)
        self.bn1b = nn.BatchNorm3d(64)
        self.activation1b = nn.ReLU(64)
        self.maxpool1 = nn.MaxPool3d(2)

        self.conv2a = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.bn2a = nn.BatchNorm3d(128)
        self.activation2a = nn.ReLU(128)
        self.conv2b = nn.Conv3d(128, 128, kernel_size=3, padding=1)
        self.bn2b = nn.BatchNorm3d(128)
        self.activation2b = nn.ReLU(128)
        self.maxpool2 = nn.MaxPool3d(2)

        self.conv3a = nn.Conv3d(128, 256, kernel_size=3, padding=1)
        self.bn3a = nn.BatchNorm3d(256)
        self.activation3a = nn.ReLU(256)
        self.conv3b = nn.Conv3d(256, 256, kernel_size=3, padding=1)
        self.bn3b = nn.BatchNorm3d(256)
        self.activation3b = nn.ReLU(256)
        self.maxpool3 = nn.MaxPool3d(2)

        self.conv4a = nn.Conv3d(256, 512, kernel_size=3, padding=1)
        self.bn4a = nn.BatchNorm3d(512)
        self.activation4a = nn.ReLU(512)
        self.conv4b = nn.Conv3d(512, 512, kernel_size=3, padding=1)
        self.bn4b = nn.BatchNorm3d(512)
        self.activation4b = nn.ReLU(512)
        self.maxpool4 = nn.MaxPool3d(2)

        self.up_conv1 = nn.ConvTranspose3d(512, 512, kernel_size=2, stride=2)
        self.conv5a = nn.Conv3d(512 + 256, 256, kernel_size=3, padding=1)
        self.bn5a = nn.BatchNorm3d(256)
        self.activation5a = nn.ReLU(256)
        self.conv5b = nn.Conv3d(256, 256, kernel_size=3, padding=1)
        self.bn5b = nn.BatchNorm3d(256)
        self.activation5b = nn.ReLU(256)

        self.up_conv2 = nn.ConvTranspose3d(256, 256, kernel_size=2, stride=2)
        self.conv6a = nn.Conv3d(256 + 128, 128, kernel_size=3, padding=1)
        self.bn6a = nn.BatchNorm3d(128)
        self.activation6a = nn.ReLU(128)
        self.conv6b = nn.Conv3d(128, 128, kernel_size=3, padding=1)
        self.bn6b = nn.BatchNorm3d(128)
        self.activation6b = nn.ReLU(128)

        self.up_conv3 = nn.ConvTranspose3d(128, 128, kernel_size=2, stride=2)
        self.conv7a = nn.Conv3d(128 + 64, 64, kernel_size=3, padding=1)
        self.bn7a = nn.BatchNorm3d(64)
        self.activation7a = nn.ReLU(64)
        self.conv7b = nn.Conv3d(64, 64, kernel_size=3, padding=1)
        self.bn7b = nn.BatchNorm3d(64)
        self.activation7b = nn.ReLU(64)

        self.finalconv = nn.Conv3d(64, n_class, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        self.skip_out64 = self.activation1b(self.bn1b(self.conv1b(self.activation1a(self.bn1a(self.conv1a(x))))))
        self.out64 = self.maxpool1(self.skip_out64)

        self.skip_out128 = self.activation2b(
            self.bn2b(self.conv2b(self.activation2a(self.bn2a(self.conv2a(self.out64))))))
        self.out128 = self.maxpool2(self.skip_out128)

        self.skip_out256 = self.activation3b(
            self.bn3b(self.conv3b(self.activation3a(self.bn3a(self.conv3a(self.out128))))))
        self.out256 = self.maxpool3(self.skip_out256)

        self.skip_out512 = self.activation4b(
            self.bn4b(self.conv4b(self.activation4a(self.bn4a(self.conv4a(self.out256))))))
        self.out512 = self.skip_out512

        self.out_up_conv1 = self.up_conv1(self.out512)
        self.concat1 = torch.cat((self.out_up_conv1, self.skip_out256), 1)
        self.out_up_256 = self.activation5b(
            self.bn5b(self.conv5b(self.activation5a(self.bn5a(self.conv5a(self.concat1))))))

        self.out_up_conv2 = self.up_conv2(self.out_up_256)
        self.concat2 = torch.cat((self.out_up_conv2, self.skip_out128), 1)
        self.out_up_128 = self.activation6b(
            self.bn6b(self.conv6b(self.activation6a(self.bn6a(self.conv6a(self.concat2))))))

        self.out_up_conv3 = self.up_conv3(self.out_up_128)
        self.concat3 = torch.cat((self.out_up_conv3, self.skip_out64), 1)
        self.out_up_64 = self.activation7b(
            self.bn7b(self.conv7b(self.activation7a(self.bn7a(self.conv7a(self.concat3))))))

        self.out = self.sigmoid(self.finalconv(self.out_up_64))

        return self.out
