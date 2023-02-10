"""

Tversky loss
"""

import torch
import torch.nn as nn


class TverskyLoss(nn.Module):
    
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        device = torch.device("cuda")
        pred = pred.squeeze(dim=1)
        # print(pred.size())
        # print(target.size())

        # weight_seg = torch.tensor([0.01, 0.3, 10]).to(device)
        # weight_seg = torch.tensor([0.05, 0.95]).to(device)
        weight_seg = torch.tensor([0.01, 1]).to(device)
        loss = nn.CrossEntropyLoss(weight=weight_seg, reduction='mean')

        output = loss(pred, target.long())

        return output

        # pred = pred.squeeze(dim=1)
        # smooth = 1
        # # print(pred.size(), pred.unique())
        # # print(target.size(), target.unique())

        # # dice系数的定义
        # dice = (pred * target).sum(dim=1).sum(dim=1).sum(dim=1) / ((pred * target).sum(dim=1).sum(dim=1).sum(dim=1) +
        #                                     0.3 * (pred * (1 - target)).sum(dim=1).sum(dim=1).sum(dim=1) + 
        #                                     0.7 * ((1 - pred) * target).sum(dim=1).sum(dim=1).sum(dim=1) + smooth)
        # # print(dice.size())
        # # 返回的是dice距离
        # # print(dice)
        # return torch.clamp((1 - dice).mean(), 0, 2)
