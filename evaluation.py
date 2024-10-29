# 定义各种评价指标
import torch
import numpy as np


def get_accuracy(output,target):
    if isinstance(output, np.ndarray):
        output = torch.from_numpy(output)
    if isinstance(target, np.ndarray):
        target = torch.from_numpy(target)
    N, C, H, W = output.size()
    total = N * C * H * W
    output = (output >= 0.5).float()
    target = (target == torch.max(target)).float()
    correct = (output == target).sum().item()
    acc = correct/total
    return acc


def get_dice(output,target,smooth=1e-6):
    if isinstance(output, np.ndarray):
        output = torch.from_numpy(output)
    if isinstance(target, np.ndarray):
        target = torch.from_numpy(target)
    intersection = output * target
    dice = (2 * intersection.sum() + smooth) / (output.sum() + target.sum() + smooth)
    return dice.item()


def get_iou(output,target):
    if isinstance(output, np.ndarray):
        output = torch.from_numpy(output)
    if isinstance(target, np.ndarray):
        target = torch.from_numpy(target)
    intersection = output * target
    union = torch.max(output, target)
    iou = (intersection.sum()) / (union.sum())
    return iou.item()


# 计算查准率pc
def get_precision(output, target):
    if isinstance(output, np.ndarray):
        output = torch.from_numpy(output)
    if isinstance(target, np.ndarray):
        target = torch.from_numpy(target)
    # TP : True Positive
    # FP : False Positive
    output = (output >= 0.5).float()
    target = (target == torch.max(target)).float()
    TP = (((output==1).float() + (target==1).float()) == 2).sum()
    FP = (((output==1).float() + (target==0).float()) == 2).sum()
    PC = TP / (TP + FP + 1e-6)
    return PC.item()

# 计算召回率（查全率）rc
def get_recall(output, target):
    if isinstance(output, np.ndarray):
        output = torch.from_numpy(output)
    if isinstance(target, np.ndarray):
        target = torch.from_numpy(target)
    # TP : True Positive
    # FN : False Negative
    # RC = TP / ((TP + FN) + 1e-6)
    output = (output >= 0.5).float()
    target = (target == torch.max(target)).float()
    TP = (((output==1).float() + (target==1).float()) == 2).sum()
    FN = (((output==0).float() + (target==1).float()) == 2).sum()
    RC = TP / (TP + FN + 1e-6)
    return RC.item()


def get_dice2(output, target):
    if isinstance(output, np.ndarray):
        output = torch.from_numpy(output)
    if isinstance(target, np.ndarray):
        target = torch.from_numpy(target)
    output = (output >= 0.5).float()
    target = (target == torch.max(target)).float()
    intersection = output * target
    dice2 = (2 * intersection.sum() + 1e-6) / (output.sum() + target.sum() + 1e-6)
    return dice2.item()