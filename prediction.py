import torch
import torchvision
import collections
import matplotlib.pyplot as plt
from dataloader import get_loader
from evaluation import get_dice,get_accuracy
from network_1 import U_Net
from network_2 import unet,SSLSE

def predict( paras_path, model="SSLSE"):

    net = SSLSE(ch_in=1, ch_out=1)
    if model == "U_Net":
        net = U_Net(img_ch=1, output_ch=1)
    elif model == "unet":
        net = unet(ch_in=1, ch_out=1)
    elif model == "FCN":
        net = torchvision.models.segmentation.fcn_resnet101(num_classes=1,progress=False)
        in_channels = 1
        net.backbone.conv1 = torch.nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2),
                                             padding=(3, 3), bias=True)

    net.eval()
    net.load_state_dict(torch.load(paras_path))

    test_dataloader = get_loader(img_root_1=r'Dataset/test/test_B/img',
                                 gt_root_1=r'Dataset/test/test_B/target',
                                 img_root_2=r'Dataset/test/test_L/img',
                                 gt_root_2=r'Dataset/test/test_L/target',
                                 mode='test')

    for idx, (img, target) in enumerate(test_dataloader):
        threshold = 0.5
        output = net(img)
        if isinstance(output, collections.OrderedDict):
            output = output['out']
            output = torch.sigmoid(output)
        output = (output >= threshold).float()
        output = output.squeeze().cpu().numpy()
        target = target.squeeze().cpu().numpy()
        img = img.squeeze().cpu().numpy()

        imgacc = []
        outacc = []
        imgdice = []
        outdice = []
        for i in range(8):
            imgx = img[i]
            outputx = output[i]
            targetx = target[i]
            imgx = (imgx >= threshold)
            outputx = (outputx >= threshold)
            targetx = targetx
            imgaccx = get_accuracy(imgx, targetx)
            imgacc.append(imgaccx)
            outaccx = get_accuracy(outputx, targetx)
            outacc.append(outaccx)
            imgdicex = get_dice(imgx, targetx)
            imgdice.append(imgdicex)
            outdicex = get_dice(outputx, targetx)
            outdice.append(outdicex)

        # 显示前四个
        fig, axes = plt.subplots(3, 4, figsize=(16, 8))  # 创建 2x4 的子图
        for i in range(4):
            ax = axes[i // 4, i % 4]  # 确定当前子图的位置
            ax.imshow(img[i], cmap='gray')  # 显示二进制掩码
            ax.axis('off')  # 关闭坐标轴
            ax.set_title(f'Input {i + 1},\nacc:{imgacc[i]:.3f},\ndice:{imgdice[i]:.3f}')
        for i in range(4):      
            ax = axes[(i // 4) + 1, i % 4]  # 确定当前子图的位置
            ax.imshow(output[i], cmap='gray')  # 显示二进制掩码
            ax.axis('off')  # 关闭坐标轴
            ax.set_title(f'Output {i + 1},\nacc:{outacc[i]:.3f},\ndice:{outdice[i]:.3f}')
        for i in range(4):
            ax = axes[(i // 4) + 2, i % 4]  # 确定当前子图的位置
            ax.imshow(target[i], cmap='gray')  # 显示二进制掩码
            ax.axis('off')  # 关闭坐标轴
            ax.set_title(f'Target {i + 1}')

        plt.tight_layout()
        plt.show()

        # 显示后四个
        fig, axes = plt.subplots(3, 4, figsize=(16, 8))
        for i in range(4):
            ax = axes[i // 4, i % 4]  # 确定当前子图的位置
            ax.imshow(img[i+4], cmap='gray')  # 显示二进制掩码
            ax.axis('off')  # 关闭坐标轴
            ax.set_title(f'Input {i + 5},\nacc:{imgacc[i+4]:.3f},\ndice:{imgdice[i+4]:.3f}')
        for i in range(4):
            ax = axes[(i // 4) + 1, i % 4]  # 确定当前子图的位置
            ax.imshow(output[i+4], cmap='gray')  # 显示二进制掩码
            ax.axis('off')  # 关闭坐标轴
            ax.set_title(f'Output {i + 5},\nacc:{outacc[i+4]:.3f},\ndice:{outdice[i+4]:.3f}')
        for i in range(4):
            ax = axes[(i // 4) + 2, i % 4]  # 确定当前子图的位置
            ax.imshow(target[i+4], cmap='gray')  # 显示二进制掩码
            ax.axis('off')  # 关闭坐标轴
            ax.set_title(f'Target {i + 5}')

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    model = "SSLSE"
    paras_path = r'SSLSE_save/model_paras.pth'
    predict(paras_path=paras_path,model=model)


