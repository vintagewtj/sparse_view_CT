import torch
from torch import nn
import os
# print(torch.cuda.is_available())                #是否有可用的gpu
# print(torch.cuda.device_count())                #有几个可用的gpu
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"        #声明gpu
device = torch.device("cuda:0")                 #调用哪个gpu

# if hasattr(torch.cuda, 'empty_cache'):
#     torch.cuda.empty_cache()
# 定义Unet模型各个模块: 卷积（特征提取）—池化（下采样）—反卷积（上采样）
class Conv1(nn.Module):
    def __init__(self, in_chs, out_chs):
        super(Conv1, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_chs, out_chs, 3, padding='same'),
            nn.BatchNorm2d(out_chs),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        x = self.conv1(x)
        return x

class Conv2(nn.Module):
    def __init__(self, in_chs, out_chs):
        super(Conv2, self).__init__()
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_chs, out_chs, 3, padding='same'),
            nn.BatchNorm2d(out_chs),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_chs, out_chs, 3, padding='same'),
            nn.BatchNorm2d(out_chs),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        x = self.conv2(x)
        return x


class Conv3(nn.Module):
    def __init__(self, in_chs, out_chs):
        super(Conv3, self).__init__()
        self.conv3 = nn.Conv2d(in_chs, out_chs, 1, padding='same')

    def forward(self, x):
        x = self.conv3(x)
        return x

class Maxpool(nn.Module):
    def __init__(self, pad):
        super(Maxpool, self).__init__()
        self.max_pool = nn.MaxPool2d(kernel_size=2, padding=pad)

    def forward(self, x):
        x = self.max_pool(x)
        return x

class Conv_trans(nn.Module):
    def __init__(self, in_chs, out_chs, kernel, pad):
        super(Conv_trans, self).__init__()
        self.conv_trans = nn.Sequential(
            nn.ConvTranspose2d(in_chs, out_chs, kernel_size=kernel, stride=2, padding=pad),
            nn.BatchNorm2d(out_chs),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        x = self.conv_trans(x)
        return x

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv0 = Conv1(1, 64)
        self.conv1 = Conv2(64, 64)
        self.down1 = Maxpool(0)                   # 第一次下采样
        self.conv2 = Conv2(64, 128)
        self.down2 = Maxpool(1)                   # 第二次下采样
        self.conv3 = Conv2(128, 256)
        self.down3 = Maxpool(1)                   # 第三次下采样
        self.conv4 = Conv2(256, 512)
        self.down4 = Maxpool(0)                   # 第四次下采样
        self.conv5 = Conv2(512, 1024)
        self.up1 = Conv_trans(1024, 512, 2, 0)    # 第一次上采样
        self.conv6 = Conv2(1024, 512)
        self.up2 = Conv_trans(512, 256, 3, 1)     # 第二次上采样
        self.conv7 = Conv2(512, 256)
        self.up3 = Conv_trans(256, 128, 3, 1)     # 第三次上采样
        self.conv8 = Conv2(256, 128)
        self.up4 = Conv_trans(128, 64, 2, 0)      # 第四次上采样
        self.conv9 = Conv2(128, 64)
        self.conv10 = Conv3(64, 1)

    def forward(self, x):
        x1 = self.conv0(x)
        x2 = self.conv1(x1)
        x3 = self.down1(x2)
        x4 = self.conv2(x3)
        x5 = self.down2(x4)
        x6 = self.conv3(x5)
        x7 = self.down3(x6)
        x8 = self.conv4(x7)
        x9 = self.down4(x8)
        y9 = self.conv5(x9)
        y8 = torch.concat((x8, self.up1(y9)), 1)      # 特征拼接(用之前相应特征层的特征进行特征“正反馈”）
        y7 = self.conv6(y8)
        y6 = torch.concat((x6, self.up2(y7)), 1)
        y5 = self.conv7(y6)
        y4 = torch.concat((x4, self.up3(y5)), 1)
        y3 = self.conv8(y4)
        y2 = torch.concat((x2, self.up4(y3)), 1)
        y1 = self.conv9(y2)
        y = self.conv10(y1)
        return y

if __name__ == '__main__':
    model = Net()
    model.to(device)
    with torch.no_grad():
        input = torch.ones(8, 1, 362, 362)
        input = input.to(device)
        output = model(input)
        output = output.to(device)
    print(output.shape)

