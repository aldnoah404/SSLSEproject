# import torchvision
# import torch

# model = torchvision.models.segmentation.deeplabv3_resnet101(progress=False, num_classes=1)
# model_modules = [m for m in model.modules()]
# print(model_modules)
# # model.backbone.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2),
# #                                              padding=(3, 3), bias=True)
# # print(model.backbone)

import torch  
from torchvision import models  

# 创建模型  
model = models.segmentation.fcn_resnet101(progress=False, num_classes=1)
model.eval()  # 设置模型为评估模式  

# 创建一个假输入（批次大小为1，3通道，输入图像大小为520x520）  
input_tensor = torch.rand(8, 3, 256, 256)  

# 前向传播  
with torch.no_grad():  
    output = model(input_tensor)  

# 查看输出  
out = output['out']  # 获取分割结果  
print(out.shape)  # 输出形状
