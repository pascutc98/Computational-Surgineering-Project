
train_ct_path = '/content/drive/MyDrive/CS/Datasets/LiTS/train/'  

train_seg_path = '/content/drive/MyDrive/CS/Datasets/LiTS/train/'  

test_ct_path = '/content/drive/MyDrive/CS/Datasets/LiTS/test/ct'

test_seg_path = '/content/drive/MyDrive/CS/Datasets/LiTS/test/seg'

training_set_path = './train/' 
 
# pred_path = '/content/drive/MyDrive/CS/Results'
# pred_path = '/content/drive/MyDrive/CS/Results/KiU-Net/Multiclass segmentation/5_48slices_transformations'  
# pred_path = '/content/drive/MyDrive/CS/Results/KiU-Net/Liver segmentation/1_48slices_transformations/'
pred_path = '/content/drive/MyDrive/CS/Results/KiU-Net/Tumor segmentation/1_48slices_transformations/'

crf_path = './crf' 

# module_path = '/content/drive/MyDrive/CS/Results/KiU-Net/Multiclass segmentation/5_48slices_transformations/net950-0.026-0.061.pth'
# module_path = '/content/drive/MyDrive/CS/Results/KiU-Net/Liver segmentation/1_48slices_transformations/net300-0.020-0.017.pth'
module_path = '/content/drive/MyDrive/CS/Results/KiU-Net/Tumor segmentation/1_48slices_transformations/net750-0.023-0.015.pth'
# module_path = '/content/drive/MyDrive/CS/Models/KiU-Net/LiTS/saved_networks/net950-0.026-0.061.pth'

size = 48   
# size = 96  # 使用48张连续切片作为网络的输入 (Using 48 consecutive slices as input to the network)

down_scale = 0.5  # 横断面降采样因子

expand_slice = 20  # 仅使用包含肝脏以及肝脏上下20张切片作为训练样本

slice_thickness = 1  # 将所有数据在z轴的spacing归一化到1mm

upper, lower = 200, -200  # CT数据灰度截断窗口

# ---------------------训练数据获取相关参数-----------------------------------


# -----------------------网络结构相关参数------------------------------------

drop_rate = 0.3  # dropout随机丢弃概率

# -----------------------网络结构相关参数------------------------------------


# ---------------------网络训练相关参数--------------------------------------

gpu = '0'  # 使用的显卡序号

Epoch = 2000

learning_rate = 1e-4

# learning_rate_decay = [500, 750]
# learning_rate_decay = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 700, 800, 900, 1000]
learning_rate_decay = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600]

alpha = 0.33  # 深度监督衰减系数

batch_size = 1

num_workers = 3

pin_memory = True

cudnn_benchmark = True

# ---------------------网络训练相关参数--------------------------------------


# ----------------------模型测试相关参数-------------------------------------

threshold = 0.5  # 阈值度阈值

stride = 12  # 滑动取样步长

maximum_hole = 5e4  # 最大的空洞面积

# ----------------------模型测试相关参数-------------------------------------


# ---------------------CRF后处理优化相关参数----------------------------------

z_expand, x_expand, y_expand = 10, 30, 30  # 根据预测结果在三个方向上的扩展数量

max_iter = 20  # CRF迭代次数

s1, s2, s3 = 1, 10, 10  # CRF高斯核参数

# ---------------------CRF后处理优化相关参数----------------------------------