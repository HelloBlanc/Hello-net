import os.path

import torch
from torch import optim
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision import transforms
from model import Network
from torch import nn

if __name__ == '__main__':
    if not os.path.exists('./results'):
        os.mkdir('./results')
    with open('./results/loss.txt', 'w') as f:
        # 图像预处理
        transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                        transforms.ToTensor()])

        # 加载数据集
        train_dataset = datasets.ImageFolder(root='./mnist_images/train', transform=transform)
        print('length of train_load:', len(train_dataset))

        # 读入小批量数据集 每个批次包含 64张图片数据
        train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
        print('length of train_loader:', len(train_loader))  # 一共分成了938个批次, 每个批次处理64个数据 938*64 = 60032 > 60000

        # 创建网络
        myNet = Network()
        # 创建优化器
        optimizer = optim.Adam(myNet.parameters())
        # 创建损失函数 --> 这里选择交叉熵损失函数
        loss_fn = nn.CrossEntropyLoss()

        # 进入模型的迭代循环
        for epoch in range(10):  # 外层循环, 代表整个训练集的训练次数
            # 内层循环使用train_loader, 进行小批量的数据读取
            for batch_idx, (data, label) in enumerate(train_loader):
                # 内层每一次循环, 都进行一次梯度下降
                # 包括了5个步骤
                output = myNet(data)  # 1.计算神经网络前向传播的结果
                loss = loss_fn(output, label)  # 2.计算output与label之间的损失
                loss.backward()  # 3.反向传播 --> 计算梯度
                optimizer.step()  # 4.梯度更新
                optimizer.zero_grad()  # 5.梯度清零

                # 每迭代100个小批次, 打印一次损失, 观察训练过程
                if batch_idx % 100 == 0:
                    print(f"Epoch {epoch + 1}/{10}, "
                          f"batch {batch_idx}/{len(train_loader)}, "
                          f"loss: {loss.item():.4f}")  # item()函数将张量转化为普通python数字类型
                    # 将损失函数写入文件
                    f.write(f"Epoch {epoch + 1}/{10}| Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}\n")
        # 保存训练好的模型
        torch.save(myNet.state_dict(), './mnist.pth')