from torchvision import datasets, transforms
from torchvision.transforms import ToPILImage
import os
import secrets # 一个专门用于生成密码学安全的随机数和密钥的标准库模块

# 加载好训练和测试需要的数据集
train_data = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_data = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor(), download=True)

print('length of train_dataset:', len(train_data))
print('length of test_dataset:', len(test_data))

"""
将数据拆分到两个文件夹中, 一个train 一个test 每个文件夹中分别有10个类别的文件夹
"""

# 将图片类型转换为张量
train_data = [(ToPILImage()(img),label) for img,label in train_data]
test_data = [(ToPILImage()(img),label) for img,label in test_data]

# 将处理好的文件保存到文件夹中
def save_images(dataset,folder_name):
    root_dir = os.path.join('./mnist_images', folder_name)
    os.makedirs(root_dir, exist_ok=True)
    for i in range(len(dataset)):
        img,label = dataset[i]
        label_dir = os.path.join(root_dir, str(label))
        os.makedirs(label_dir, exist_ok=True)
        random_filename = secrets.token_hex(8) + '.png'
        img.save(os.path.join(label_dir, random_filename))

save_images(train_data,'train')
save_images(test_data,'test')
print('finish!!')