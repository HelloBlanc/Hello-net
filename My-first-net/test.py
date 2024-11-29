from torchvision import datasets,transforms
from model import Network
import torch

if __name__ == '__main__':
    # 图像预处理 --> 将图片数据转换成张量类型
    transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                    transforms.ToTensor()])
    # 加载测试数据
    test_dataset = datasets.ImageFolder(root='./mnist_images/test', transform=transform)

    # 定义神经网络模型
    model = Network()
    # 导入保存好的模型
    model.load_state_dict(torch.load('mnist.pth'))

    right = 0
    for batch_id,(img,label) in enumerate(test_dataset):
        # 将img数据输入到模型中
        output = model(img)
        # 选择概率最大标签的作为预测结果
        predict = output.argmax(1).item()
        if predict == label:
            right+=1
        else:
            # 将错误识别的图片找出来
            img_path = test_dataset.samples[batch_id][0]
            print(f"wrong case: predict = {predict} label = {label} img_path = {img_path}")


    # 计算出测试结果
    sample_num = len(test_dataset)
    acc = right * 1.0 / sample_num
    print("test accuracy: %d / %d = %.3lf" % (right, sample_num, acc))




