"""

torch中的Dataset定义脚本
"""

import os
import sys
sys.path.append(os.path.split(sys.path[0])[0])

import random

import numpy as np
import SimpleITK as sitk

import torch
from torch.utils.data import Dataset as dataset
import torchvision.transforms as transforms

import parameter as para

def add_gaussian_noise(img, mean=0, std=0.01, p=0.05):
    if random.uniform(0, 1) < p:
        noise = torch.randn(img.size(), dtype=torch.float32) * std + mean
        img = img + noise
    return img


class Dataset(dataset):
    def __init__(self, ct_dir, seg_dir, apply_transforms):
        self.ct_list = os.listdir(ct_dir)
        self.seg_list = list(map(lambda x: x.replace('volume', 'segmentation').replace('.nii', '.nii.gz'), self.ct_list))

        self.ct_list = list(map(lambda x: os.path.join(ct_dir, x), self.ct_list))
        self.seg_list = list(map(lambda x: os.path.join(seg_dir, x), self.seg_list))

        self.apply_transforms = apply_transforms

        # Initialize the data augmentation transforms here
        self.data_augmentation_1 = transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomAffine(0, translate=(0.3, 0.3))
        ])

        self.data_augmentation_2 = transforms.Compose([
            transforms.RandomHorizontalFlip(p=1),
            transforms.RandomRotation(30),
            transforms.RandomAffine(0, translate=(0.3, 0.3))
        ])


    def __getitem__(self, index):

        ct_path = self.ct_list[index]
        seg_path = self.seg_list[index]
        indices_liver = []

        # 将CT和金标准读入到内存中
        ct = sitk.ReadImage(ct_path, sitk.sitkInt16)
        seg = sitk.ReadImage(seg_path, sitk.sitkUInt8)

        ct_array = sitk.GetArrayFromImage(ct)
        seg_array = sitk.GetArrayFromImage(seg)

        seg_array[seg_array > 2] = 1 # Apply in general
        # seg_array[seg_array > 0] = 1 # Only liver detection
        seg_array[seg_array == 1] = 0 # Only tumor detection
        seg_array[seg_array == 2] = 1 # Only tumor detection.
        
        # min max 归一化
        ct_array = ct_array.astype(np.float32)
        ct_array[ct_array > para.upper] = para.upper
        ct_array[ct_array < para.lower] = para.lower
        ct_array = ct_array / 200

        # Liver label detection
        # for j in range(np.shape(seg_array)[0]):
        #   if np.count_nonzero(seg_array[j,:,:] == 1) > 0:
        #       indices_liver.append(j)

        # 在slice平面内随机选取48张slice
        start_slice = random.randint(0, ct_array.shape[0] - para.size)
        # value = round((indices_liver[-1] - indices_liver[0])/2)
        # start_slice = random.randint(indices_liver[0], value)
        end_slice = start_slice + para.size - 1

        ct_array = ct_array[start_slice:end_slice + 1, :, :]
        seg_array = seg_array[start_slice:end_slice + 1, :, :] 

        # 处理完毕，将array转换为tensor
        ct_array = torch.FloatTensor(ct_array)
        seg_array = torch.FloatTensor(seg_array)


        # Now apply data augmentation to the images
        if self.apply_transforms == 1:
            images = torch.cat((ct_array, seg_array), dim=0)
            images = self.data_augmentation_1(images)
            ct_array, seg_array = torch.split(images, [ct_array.shape[0], seg_array.shape[0]], dim=0)
        elif self.apply_transforms == 2:
            images = torch.cat((ct_array, seg_array), dim=0)
            images = self.data_augmentation_2(images)
            ct_array, seg_array = torch.split(images, [ct_array.shape[0], seg_array.shape[0]], dim=0)

        ct_array = ct_array.unsqueeze(0)
        #print(ct_array.size(), seg_array.size())
        return ct_array, seg_array

    def __len__(self):

        return len(self.ct_list)