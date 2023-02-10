"""

测试脚本
"""

import os
import copy
import collections
from time import time

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import scipy.ndimage as ndimage
import SimpleITK as sitk
import skimage.measure as measure
import skimage.morphology as morphology
from loss.DiceLoss import dice_loss


from net.models import *
from utilities.calculate_metrics import Metirc

import parameter as para

os.environ['CUDA_VISIBLE_DEVICES'] = para.gpu

# 为了计算dice_global定义的两个变量
dice_global_0 = []
dice_global_1 = []
dice_global_2 = []
dict_missing_labels = {'0':0, '1':0, '2':0}



file_name = []  # 文件名称
time_pre_case = []  # 单例数据消耗时间

net = torch.nn.DataParallel(kiunet_org(training=False)).cuda()
net.load_state_dict(torch.load(para.module_path))
net.eval()
three_classes = False


for file_index, file in enumerate(os.listdir(para.test_ct_path)):

    start = time()

    file_name.append(file)

    # 将CT读入内存
    ct = sitk.ReadImage(os.path.join(para.test_ct_path, file), sitk.sitkInt16)
    ct_array = sitk.GetArrayFromImage(ct)

    origin_shape = ct_array.shape
    
    # 将灰度值在阈值之外的截断掉
    ct_array[ct_array > para.upper] = para.upper
    ct_array[ct_array < para.lower] = para.lower

    # min max 归一化
    ct_array = ct_array.astype(np.float32)
    ct_array = ct_array / 200


    # 对slice过少的数据使用padding
    too_small = False
    if ct_array.shape[0] < para.size:
        temp = np.zeros((para.size-ct_array.shape[0],256,256))
        ct_array = np.vstack([ct_array, temp])
        too_small = True

    # 滑动窗口取样预测
    start_slice = 0
    end_slice = start_slice + para.size
    # count = np.zeros((ct_array.shape[0], 256, 256), dtype=np.int16)
    probability_map = np.zeros((ct_array.shape[0], 256, 256), dtype=np.float32)
    out = False

    with torch.no_grad():
        while end_slice < ct_array.shape[0]:

            ct_tensor = torch.FloatTensor(ct_array[start_slice: end_slice + 1,:,:]).cuda()

            ct_tensor = ct_tensor.unsqueeze(dim=0).unsqueeze(dim=0)
            outputs = net(ct_tensor)
            outputs = torch.argmax(outputs, dim=1)
            # print(outputs.size())

            probability_map[start_slice: end_slice,:,:] = np.squeeze(outputs.cpu().detach().numpy())

            # 由于显存不足，这里直接保留ndarray数据，并在保存之后直接销毁计算图
            del outputs      
            
            start_slice += para.stride
            end_slice = start_slice + para.size
            #print(start_slice, end_slice)
    
        if end_slice != ct_array.shape[0] or too_small == True:
            end_slice = ct_array.shape[0]
            start_slice = end_slice - para.size


            ct_tensor = torch.FloatTensor(ct_array[start_slice: end_slice + 1]).cuda()

            ct_tensor = ct_tensor.unsqueeze(dim=0).unsqueeze(dim=0)
            outputs = net(ct_tensor)
            number_classes = outputs.size()[1]

            outputs = torch.argmax(outputs, dim=1)

            probability_map[start_slice: end_slice,:,:] = np.squeeze(outputs.cpu().detach().numpy())

            del outputs
        
        pred_seg = probability_map


    # 将金标准读入内存
    seg = sitk.ReadImage(os.path.join(para.test_seg_path, file.replace('volume', 'segmentation')), sitk.sitkUInt8)
    seg_array = sitk.GetArrayFromImage(seg)
    seg_array[seg_array > 2] = 1 # Apply in general
    # seg_array[seg_array > 0] = 1 # Only liver detection
    seg_array[seg_array == 1] = 0 # Only tumor detection
    seg_array[seg_array >= 2] = 1 # Only tumor detection.

    if too_small:
      temp = np.zeros((para.size-seg_array.shape[0],256,256))
      seg_array = np.vstack([seg_array, temp])


    # 对肝脏进行最大连通域提取,移除细小区域,并进行内部的空洞填充 (Maximum connectivity extraction of the liver, removal of fine areas, and internal cavity filling)
    pred_seg = pred_seg.astype(np.uint8)
    liver_seg = copy.deepcopy(pred_seg)

    liver_seg = liver_seg.astype(np.uint8)
    seg_array = seg_array.astype(np.uint8)

    print(f"Number of classes: {number_classes}")
    print(f"Prediction (unique classes): {np.unique(liver_seg)} // Prediction (shape) {np.shape(liver_seg)}")
    print(f"Target (unique classes): {np.unique(seg_array)} // Target (shape) {np.shape(seg_array)}")

    dice = dice_loss(seg_array, liver_seg)
    print(dice)

    for key in dice.keys():
      if isinstance(dice[key], str):
        dict_missing_labels[key] += 1
        continue
      elif key == '0':
        dice_global_0.append(dice[key])
      elif key == '1':
        dice_global_1.append(dice[key])
      elif key == '2':
        dice_global_2.append(dice[key])

    # 将预测的结果保存为nii数据
    pred_seg = sitk.GetImageFromArray(liver_seg)

    pred_seg.SetDirection(ct.GetDirection())
    pred_seg.SetOrigin(ct.GetOrigin())
    pred_seg.SetSpacing(ct.GetSpacing())

    sitk.WriteImage(pred_seg, os.path.join(para.pred_path, file.replace('volume', 'pred')))

    speed = time() - start
    time_pre_case.append(speed)



    print(file, 'this case use {:.3f} s'.format(speed))
    print('-----------------------')


# 将评价指标写入到exel中
# df = liver_data = pd.DataFrame({'0': np.mean(dice_global_0), '1': np.mean(dice_global_1), '2': np.mean(dice_global_2)})
# df.to_excel(os.path.join(para.pred_path, "data.xlsx"), index=False)



# liver_data['time'] = time_pre_case

# liver_statistics = pd.DataFrame(index=['mean', 'std', 'min', 'max'], columns=list(liver_data.columns))
# liver_statistics.loc['mean'] = liver_data.mean()
# liver_statistics.loc['std'] = liver_data.std()
# liver_statistics.loc['min'] = liver_data.min()
# liver_statistics.loc['max'] = liver_data.max()

# writer = pd.ExcelWriter('./result.xlsx')
# liver_data.to_excel(writer, 'liver')
# liver_statistics.to_excel(writer, 'liver_statistics')
# writer.save()

# 打印dice global
print(f"Number of test patients: {len(os.listdir(para.test_ct_path))}")
print(f"Missing labels -> {dict_missing_labels}")
print(f"Dice global loss (label 0) -> {np.mean(dice_global_0)}")
print(f"Dice global loss (label 1) -> {np.mean(dice_global_1)}")
print(f"Dice global loss (label 2) -> {np.mean(dice_global_2)}")
