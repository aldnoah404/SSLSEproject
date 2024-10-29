import collections
import os
import torchvision
import torch
import numpy as np
from evaluation import *
from dataloader import get_loader,rm_mkdir
from DicexFocal_Loss import DiceFocal_Loss
from torch.optim import Adam,lr_scheduler
from network_1 import U_Net
from network_2 import unet,init_weights,SSLSE
from thop import profile


def main(model="SSLSE", epochs=200, retrain=False):
    # 设置装置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device_info:{device}')
    # 设置模型保存路径
    if retrain:
        rm_mkdir(f'{model}_save')
        print("————————重新开始训练————————")
    if not os.path.exists(f'{model}_save'):
        os.makedirs(f'{model}_save')
    # 加载数据
    train_dataloader = get_loader(img_root_1=r'Dataset/train/train_B/img',
                                  gt_root_1=r'Dataset/train/train_B/target',
                                  img_root_2=r'Dataset/train/train_L/img',
                                  gt_root_2=r'Dataset/train/train_L/target',
                                  mode='train')
    test_dataloader = get_loader(img_root_1=r'Dataset/test/test_B/img',
                                 gt_root_1=r'Dataset/test/test_B/target',
                                 img_root_2=r'Dataset/test/test_L/img',
                                 gt_root_2=r'Dataset/test/test_L/target',
                                 mode='test')

    # 设置模型，损失函数，优化器
    net = SSLSE(ch_in=1, ch_out=1)
    if model == "U_Net":
        net = U_Net(img_ch=1, output_ch=1)
    elif model == "unet":
        net = unet(ch_in=1, ch_out=1)
    elif model == "FCN":
        # net = torchvision.models.segmentation.
        net = torchvision.models.segmentation.fcn.FCN(num_classes=1, progress=False)
        # 修改输入层
        # 用 nn.Conv2d 替换第一个卷积层
        in_channels = 1  
        net.backbone.conv1 = torch.nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2),
                                             padding=(3, 3), bias=True)
    elif model == "DeeplabV3":
        net = torchvision.models.segmentation.deeplabv3_resnet101(progress=False, num_classes=1)
        # 修改输入层
        # 用 nn.Conv2d 替换第一个卷积层
        in_channels = 1  
        net.backbone.conv1 = torch.nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2),
                                             padding=(3, 3), bias=True)
    
    net = net.to(device=device)
    print(f'Model_info:{model}')
    input1 = torch.rand([8, 1, 288, 288])
    input1 = input1.to(device=device)
    flops, params = profile(net, inputs=(input1,))
    print(f'——————>FLOPs = {flops / 1000 ** 3:.3f} G')
    print(f'——————>params = {params / 1000 ** 2:.3f} M')

    # 初始化模型/加载参数
    paras_path = f'./{model}_save/model_paras.pth'
    if os.path.isfile(paras_path):
        net.load_state_dict(torch.load(paras_path))
        print(f'模型加载路径：———— {paras_path}')
    else:
        init_weights(net=net)

    # 实例化损失函数
    Loss_func = DiceFocal_Loss()
    Loss_func = Loss_func.to(device=device)

    # 实例化优化器和设置学习率策略
    optimizer = Adam(net.parameters(), lr=1e-4)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=64, eta_min=1e-5)

    # 设置轮次
    epochs = epochs
    flag = 0
    epoch_flag = None

    # 开始训练
    net.train()
    for epoch in range(epochs):
        print(f'————————epoch:{epoch}————————')
        for idx, (img, target) in enumerate(train_dataloader):
            img = img.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            output = net(img)
            if isinstance(output, collections.OrderedDict):
                output = output['out']
                output = torch.sigmoid(output)

            result_loss = Loss_func(output, target)
            result_loss.backward()
            optimizer.step()

            if (idx + 1) % 25 == 0:
                acc = get_accuracy(output, target)
                dice = get_dice(output, target)
                dice2 = get_dice2(output, target)
                iou = get_iou(output, target)
                precession = get_precision(output, target)
                recall = get_recall(output, target)
                result_loss = result_loss.item()
                print(f'idx：{idx + 1} ————>loss  :{result_loss:.5f}\n\t'
                        f'————>acc   :{acc:.5f}\n\t'
                        f'————>dice  :{dice:.5f}\n\t'
                        f'————>dice2 :{dice2:.5f}\n\t'
                        f'————>iou   :{iou:.5f}\n\t'
                        f'————>pc    :{precession:.5f}\n\t'
                        f'————>rc    :{recall:.5f}')
        scheduler.step()

        # 保存模型数据
        if (epoch+1) % 10 == 0:
            save_path = f'{model}_save/epoch{epoch+1}_paras.pth'
            torch.save(net.state_dict(), save_path)

        # 测试性能
        net.eval()
        accs = []
        dices = []
        dice2s = []
        ious = []
        precessions = []
        recalls = []
        losses = []
        with torch.no_grad():
            for idx, (img, target) in enumerate(test_dataloader):
                img = img.to(device)
                target = target.to(device)
                output = net(img)
                if isinstance(output, collections.OrderedDict):
                    output = output['out']
                    output = torch.sigmoid(output)
                target = target.cpu()
                output = output.cpu()

                result_acc = get_accuracy(output=output, target=target)
                result_dice = get_dice(output=output, target=target)
                result_dice2 = get_dice2(output, target)
                result_iou = get_iou(output, target)
                result_precession = get_precision(output, target)
                result_recall = get_recall(output, target)
                result_loss = Loss_func(output, target).item()

                losses.append(result_loss)
                accs.append(result_acc)
                dices.append(result_dice)
                dice2s.append(result_dice2)
                ious.append(result_iou)
                precessions.append(result_precession)
                recalls.append(result_recall)

            acc = np.array(accs).mean()
            dice = np.array(dices).mean()
            dice2 = np.array(dice2s).mean()
            precession = np.array(precessions).mean()
            recall = np.array(recalls).mean()
            iou = np.array(ious).mean()
            loss = np.array(losses).mean()

            print(f'————————test————————')
            print(f' ————>loss  :{loss:.5f}\n\t'
                        f'————>acc   :{acc:.5f}\n\t'
                        f'————>dice  :{dice:.5f}\n\t'
                        f'————>dice2 :{dice2:.5f}\n\t'
                        f'————>iou   :{iou:.5f}\n\t'
                        f'————>pc    :{precession:.5f}\n\t'
                        f'————>rc    :{recall:.5f}')
            result_save_path = f'{model}_save/result.txt'
            with open(result_save_path,'a',encoding='utf-8') as result_file:
                print(f'{epoch:0<4},{acc:.5f},{dice:.5f},{iou:.5f}',file=result_file)

            if (acc+dice) > flag:
                flag = (acc+dice)
                epoch_flag = epoch+1
                # save_path = f'./model_save/model_paras.pth'
                save_path = f'{model}_save/model_paras.pth'
                torch.save(net.state_dict(), save_path)
            print(f'epoch_flag:{epoch_flag}\n')
    with open(result_save_path,'a',encoding='utf-8') as result_file:
        print(f'best_epoch:{epoch_flag}')


if __name__ == '__main__':
    models = ["SSLSE", "U_Net", "unet", "FCN", "DeeplabV3"]
    print("可供选择的模型包括：")
    print("\t1.SSLSE")
    print("\t2.U_Net")
    print("\t3.unet")
    print("\t4.FCN(fcn_resnet101)")
    print("\t5.DeeplabV3(deeplabv3_resnet101)")
    print()
    print()
    num = int(input("请选择要使用的模型：")) - 1
    model = models[num]
    main(model=model,epochs=200, retrain=True)



