import torch
import torchvision
from torch import nn, cuda
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import time
from model import *       # import model 的话创建网络就要写成 model = model.Net()
from dataset_generator import *
import pycuda.autoinit
# cfx = cuda.Device(0).make_context()
# src = torch.cuda.ByteTensor(8)# 数字随便填写，矩阵也可以，但是数据类型要和C++中的数据类型保持一致
# b = torch.cuda.FloatTensor(9)


# 准备数据集
train_data = Datasets('../original_data/LDCT_DATA/ground_truth_train')
test_data = Datasets('../original_data/LDCT_DATA/ground_truth_test')

# 查看数据集大小 length函数
train_data_size = len(train_data)
test_data_size = len(test_data)
print("训练数据集的长度为:{}".format(train_data_size))  # 字符串的格式化使用
print("测试数据集的长度为:{}".format(test_data_size))

# 利用dataloader 来加载数据集
train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=4)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True, num_workers=4)

if __name__ == '__main__':
    # 创建网络模型
    model = Net()
    # cfx.push()
    device = torch.device("cuda:0")
    model.to(device)

    # 损失函数
    loss_fn = nn.MSELoss(reduction='mean')
    loss_fn.to(device)

    # 优化器
    learning_rate = 1e-2  # learning_rate = 0.01   1e-2 = (10)^(-2) = 0.01
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, )

    # 设置训练网络的一些参数
    total_train_step = 0  # 记录训练的次数
    total_test_step = 0  # 记录测试的次数
    epoch = 20  # 训练的轮数

    # 添加tensorboard
    writer = SummaryWriter("../logs_train")

    # 训练步骤开始
    model.train()
    start_time = time.time()
    for i in range(epoch):
        print("----------第 {} 轮训练开始----------".format(i + 1))
        for data in train_dataloader:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = model(imgs)
            loss = loss_fn(outputs, targets)

            # 优化器优化模型
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_step += 1
            if total_train_step % 100 == 0:
                end_time = time.time()
                print("训练时间：{}".format(end_time - start_time))
                print("训练次数: {0}，Loss: {1}".format(total_train_step, loss.item()))
                writer.add_scalar("train_loss", loss.item(), total_train_step)

        # 测试步骤开始
        model.eval()
        total_test_loss = 0
        total_accuracy = 0
        with torch.no_grad():  # 查看模型训练时不生成计算图
            for data in test_dataloader:
                imgs, targets = data
                imgs = imgs.to(device)
                targets = targets.to(device)
                outputs = model(imgs)
                loss = loss_fn(outputs, targets)
                total_test_loss += loss.item()

        print("整体测试集上的Loss: {}".format(total_test_loss))
        writer.add_scalar("test_loss", total_test_loss, total_train_step)
        total_test_step += 1

        torch.save(model, "../checkpoints/model_{}.pth".format(i))
        # 官方推荐模型保存方式： torch.save(model.state_dict(), "./pth/model_{}.pth".format(i))
        print("模型已保存")

    # cfx.pop()
    writer.close()

