import torch

from cnn import simpleNet

x = torch.randn(32,3,224,224)
# 实例化一个网络
net = simpleNet(num_classes=4)
output = net(x)
print(output.size())