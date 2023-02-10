
import os
from time import time

import numpy as np

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# from visdom import Visdom

from dataset.dataset import Dataset

from loss.Dice import DiceLoss
from loss.ELDice import ELDiceLoss
from loss.WBCE import WCELoss
from loss.Jaccard import JaccardLoss
from loss.SS import SSLoss
from loss.Tversky import TverskyLoss
from loss.Hybrid import HybridLoss
from loss.BCE import BCELoss
from loss.DiceLoss import dice_loss

from net.models import net

import parameter as para

import matplotlib.pyplot as plt



# GPU
os.environ['CUDA_VISIBLE_DEVICES'] = para.gpu
cudnn.benchmark = para.cudnn_benchmark

# Network
net = torch.nn.DataParallel(net).cuda()
net.train()

# net = torch.nn.DataParallel(net).cuda()
# net.load_state_dict(torch.load(para.module_path))
# net.train()

# Path
print(para.training_set_path)

# Create DataLoader for the original data
train_ds_original = Dataset(os.path.join(para.train_ct_path, 'ct'), os.path.join(para.train_seg_path, 'seg'), apply_transforms=0)

# Create DataLoader for the transformed data (rotation, affine and random noise)
# train_ds_transformed_1 = Dataset(os.path.join(para.train_ct_path, 'ct'), os.path.join(para.train_seg_path, 'seg'), apply_transforms=1)

# Create DataLoader for the transformed data (horizontal flip, rotation, affine and random noise)
train_ds_transformed_2 = Dataset(os.path.join(para.train_ct_path, 'ct'), os.path.join(para.train_seg_path, 'seg'), apply_transforms=2)

# Combine the three DataLoader instances to form a single dataset and shuffle them
train_dl = torch.utils.data.ConcatDataset([train_ds_original, train_ds_transformed_2])
train_dl = DataLoader(train_dl, para.batch_size, shuffle=True, num_workers=para.num_workers, pin_memory=para.pin_memory)

# Loss functions (only is used TverskyLoss())
loss_func_list = [DiceLoss(), ELDiceLoss(), WCELoss(), JaccardLoss(), SSLoss(), TverskyLoss(), HybridLoss(), BCELoss()]
loss_func = loss_func_list[5]

# Optimizer
opt = torch.optim.Adam(net.parameters(), lr=para.learning_rate)

# Learning rate decay
lr_decay = torch.optim.lr_scheduler.MultiStepLR(opt, para.learning_rate_decay)

# Alpha value for the loss function calculation
alpha = para.alpha

# Tensorboard
writer = SummaryWriter('logs')

# Start timer
start = time()

for epoch in range(para.Epoch):
    lr_decay.step()
    mean_loss = []


    for step, (ct, seg) in enumerate(train_dl):

        ct = ct.cuda()
        seg = seg.cuda()

        outputs = net(ct)
        loss1 = loss_func(outputs[0], seg)
        loss2 = loss_func(outputs[1], seg)
        loss3 = loss_func(outputs[2], seg)
        loss4 = loss_func(outputs[3], seg)

        loss =  (torch.sum(loss1 + loss2 + loss3) * alpha) + loss4 # Loss function

    
        mean_loss.append(loss4.item())

        writer.add_scalar('Loss/train', loss4.item(), epoch) #Tensorboard

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 5 == 0:
            print('epoch:{}, step:{}, loss1:{:.3f}, loss2:{:.3f}, loss3:{:.3f}, loss4:{:.3f}, loss:{} time:{:.3f} min'
                  .format(epoch, step, loss1.item(), loss2.item(), loss3.item(), loss4.item(), loss, (time() - start) / 60))
            dice_loss_value = dice_loss(seg, outputs[3]) 
            print(f"Dice loss: {dice_loss_value}")

    mean_loss = sum(mean_loss) / len(mean_loss)


    print(f"Epoch: {epoch} -> Mean loss: {mean_loss}")
    print(f"Learning rate: {opt.param_groups[0]['lr']}")

    if epoch % 50 == 0 :
        torch.save(net.state_dict(), '/content/drive/MyDrive/CS/Models/KiU-Net/LiTS/saved_networks/net{}-{:.3f}-{:.3f}.pth'.format(epoch, loss, mean_loss))
    
    if epoch % 40 == 0 and epoch != 0:
        alpha *= 0.8

writer.close()


