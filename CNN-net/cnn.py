import torch.nn as nn

class simpleNet(nn.Module):
    def __init__(self,num_classes): # num_classes表示分类数
        super(simpleNet, self).__init__()
        # 用于特征提取
        self.features = nn.Sequential(
            nn.Conv2d(3,16,kernel_size=3,stride=1,padding=1),   # 将3通道数 --> 16通道数 并保持图像大小不变 16 * 224 *224
            nn.ReLU(),   # 卷积后添加激活函数, 增加非线性特征
            nn.MaxPool2d(2,stride=2), # 池化后变成 16 * 112 * 112
            nn.Conv2d(16,32,kernel_size=3,stride=1,padding=1), # 保持图像大小不变 32 * 112 * 112
            nn.ReLU(),
            nn.MaxPool2d(2,stride=2) # 图像大小变为 32 * 56 * 56
        )

        # 用于分类
        self.classifier = nn.Sequential(
            nn.Linear(32*56*56,128),
            nn.ReLU(),
            nn.Linear(128,num_classes)
        )

    # 前向传播
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x