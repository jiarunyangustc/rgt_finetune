import os
import math
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
import time
from skimage import measure
from scipy.interpolate import interp1d, interp2d, griddata
from scipy.ndimage import gaussian_filter
import torch.nn.functional as F
from PIL import Image
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau,CosineAnnealingWarmRestarts,CosineAnnealingLR,MultiStepLR,LambdaLR
from torch.utils.data import Dataset, DataLoader



from cigfaciesloss import CIGLoss, NormalLoss, SegmentLoss, SegmentOrderLoss

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors

import loralib as lora
from draw import *
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots

# from adbinloss import SILogLoss, BinsChamferLoss

from ssim import SSIMLoss as ssim
from ssim import MultiScaleSSIMLoss as mssim
from ssim import MultiScaleSSIMLoss3d as mssim_3d
from scipy.ndimage import gaussian_filter


from torch.optim.lr_scheduler import _LRScheduler
# from pytorch_msssim import ms_ssim3d
# from timm.utils import NativeScaler
# from warmup_scheduler import GradualWarmupScheduler
# from torch.optim.lr_scheduler import StepLR
# import torch.optim as optim
# from losses import CharbonnierLoss


def compute_error(output, target):
    abs_diff = np.abs(output - target)
    mse = float((np.power(abs_diff, 2)).mean())
    rmse = math.sqrt(mse)                                                                                                                       
    mae = float(abs_diff.mean())
    return mse, mae

def compute_hrzs_error(samples):
    merror = np.zeros(len(samples))
    for i in range(len(samples)):
        pred = samples[i]['pred'].squeeze()
        frame = samples[i]['frame'].squeeze()

        hvs, hms = separate_hrzs(frame, pred, bit, sample_rate)
        hrzs_f = compute_in_hrzs(frame, hms)
        hrzs_p = compute_out_hrzs(pred, hvs)

        hrz_error = []
        for hrz_f in hrzs_f:
            x_f, y_f = hrz_f
            e_f = np.ones(len(y_f)) * 1e16
            for hrz_p in hrzs_p:
                x_p, y_p = hrz_p
                for j, y in enumerate(y_f):
                    ix = np.where(y_p == y)[0]
                    if len(ix):
                        tmp = abs(x_p[ix[0]] - x_f[j])
                        if tmp >= 6.0:
                            tmp = 1e16
                        e_f[j] = min(tmp, e_f[j])

            e, c = 0, 0
            for k in range(len(e_f)):
                if e_f[k] < 6.0:
                    e += e_f[k]
                    c += 1  
            if c > 0:
                hrz_error.append(e/c)
        
        merror[i] = np.array(hrz_error).mean()
        print(f"样本{i}误差:{merror[i]}")
    print(f"平均误差:{merror.mean()}") 
    return merror



def min_max_norm_per_sample(x):
    # x.shape = [N, C, H, W] -> [10, 1, 512, 512]
    
    # 1. 计算每个样本的最小值和最大值
    # dim=(1, 2, 3) 表示聚合 Channel, Height, Width
    # keepdim=True 保持形状为 [10, 1, 1, 1]，方便后续广播
    min_val = x.amin(dim=(1, 2, 3), keepdim=True)
    max_val = x.amax(dim=(1, 2, 3), keepdim=True)
    
    # 2. 计算分母 (最大值 - 最小值)
    # 加上 1e-8 是为了防止 max == min (纯色图像) 导致除以 0
    delta = max_val - min_val + 1e-8
    
    # 3. 应用公式
    x_norm = (x - min_val) / delta
    
    return x_norm

def get_rgt_init_from_rx_field_data(fx,mx,sample_width,rgt_values):
    
    f1 = np.zeros(mx.shape)
    f2 = np.zeros(mx.shape,dtype = np.single)
    mx_single = np.zeros(mx.shape,dtype = np.single)
#     mx_single_all = np.zeros(mx.shape[1:],dtype = np.single)
    fxrm = np.zeros(mx.shape[1:])
    rgtrm = np.zeros(mx.shape[1:])
    rgt_hr = np.zeros(sample_width)
    fr_mean = np.zeros(mx.shape[1:])
    for i in range(mx.shape[0]):

        f1[i] = mx[i,]*fx

        for j in range(mx.shape[2]-1):
            for k in range(mx.shape[1]-1):
                if f1[i,k-1,j] <0.0000001 and f1[i,k,j] >0.00001:
                    f2[i,k,j] = f1[i,k,j]
                    mx_single[i,k,j] =1
#                     mx_single_all[k,j] = 1
                    break
        
        x_all,y_all =np.where(f2[i,:,:] !=0)
        
        if len(x_all) ==0:
            continue
        fr_mean += np.int64(f2[i,:,:]!=0)*rgt_values[i]
        for l in range(sample_width):
            rgt_hr[l] = rgt_values[i]+((len(x_all)*(l-int(sample_width/2)))/(len(x_all)*(127)))

        for k in range(len(x_all)):
            for j in range(sample_width):
                if 0 <= x_all[k]-int(sample_width/2)+j<mx.shape[1]:
                    fxrm[x_all[k]-int(sample_width/2)+j,y_all[k]] = rgt_hr[j]
    fr_sp = f2.sum(axis=(0))
    x,y = np.where(fxrm!=0)
    y_r = sorted(np.unique(y))
    range_x=range(128)
    for i in range(len(y_r)):
        x_r = np.where(fxrm[:,y_r[i]]!=0)

        f = interp1d(x_r[0], fxrm[x_r[0],y_r[i]], fill_value='extrapolate')
#         for j in range(128):
        rgtrm[range_x,y_r[i]] = f(range_x)

    return rgtrm,fxrm,fr_sp,fr_mean,mx_single


# 阈值压制
def threshold_filter(predictions, thr=0.01):
    thresholded_preds = predictions[:]
    low_values_indices = thresholded_preds < thr
    thresholded_preds[low_values_indices] = 0
    low_values_indices = thresholded_preds >= thr
    thresholded_preds[low_values_indices] = 1
    return thresholded_preds

# 归一化
def assign_min_max_norm(x, m, a):
    x = (x - m) / (a - m)        
    return x

def remove_min_max_norm(x, m, a):
    x = x * (a - m) + m        
    return x

def min_max_norm(x):
    if torch.is_tensor(x) and torch.max(x) != torch.min(x):
            x = x - torch.min(x)
            x = x / torch.max(x)        
    elif np.max(x) != np.min(x):
            x = x - np.min(x)
            x = x / np.max(x)
    return x
    
# 标准化
def mea_std_norm(x):
    if torch.is_tensor(x) and torch.std(x) != 0:
            x = (x - torch.mean(x)) / torch.std(x)
    elif np.std(x) != 0:
            x = (x - np.mean(x)) / np.std(x)
    return x

class SSIMLoss(nn.Module):
    def __init__(self, channel, filter_size):
        super(SSIMLoss, self).__init__()
        self.ssim = mssim(channel=channel, filter_size=filter_size)
    def forward(self, output, target, mask=None):
        loss = (1 - self.ssim(output, target, mask))
        return loss

class SSIMLoss3d(nn.Module):
    def __init__(self, channel, filter_size):
        super().__init__()
        self.ssim = mssim_3d(channel=channel, filter_size=filter_size)
    def forward(self, output, target, mask=None):
        loss = (1 - self.ssim(output, target, mask))
        return loss

# class SSIMLoss3d(nn.Module):
#     def __init__(self, channel, filter_size):
#         super().__init__()  # 推荐这种写法
#         self.ssim = mssim_3d(channel=channel, filter_size=filter_size)

#     def forward(self, x, y):
#         return 1 - self.ssim(x, y)

class MMSESSIMLoss(nn.Module):
    def __init__(self, channel, filter_size):
        super(MMSESSIMLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")
        self.ssim = mssim(channel=channel, filter_size=filter_size)
    def forward(self, output, target, mask=None):
        loss = 1 - self.ssim(output, target)
        return loss


class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()
        self.mse = nn.MSELoss(reduction="mean")
    def forward(self, output, target, mask=None):
        loss = self.mse(output, target)
        return loss

# class hr_loss(nn.Module):
#     def __init__(self):
#         super(hr_loss, self).__init__()
#         self.name = 'hr_loss'
 
#     def forward(self, mx_single, pred):
#         pred_fr = mx_single * pred
#         total_error = 0.0
#         total_count = 0

#         for i in range(mx_single.shape[0]):
#             y_mean = torch.sum(pred_fr[i]) / torch.sum(mx_single[i])
#             mx_c_i = mx_single[i] * y_mean
#             error = torch.abs(pred_fr[i] - mx_c_i)
#             total_error += torch.sum(error)
#             total_count += error.numel()

#         # 计算总的平均误差
#         mean_error = total_error / total_count
#         return mean_error
# class hr_loss(nn.Module):
#     def __init__(self):
#         super(hr_loss,self).__init__()
#         self.name = 'hr_loss'
# #         self.js = 0
# #         self.ls_sum = 0

#     def forward(self,fx_mean,pred):
# #         pred = nn.functional.interpolate(pred, mx.shape[-2:], mode='bilinear', align_corners=True).to(pred.device)
#         print(fx_mean.shape,pred.shape)
#         x,_,_,_ = torch.where(fx_mean!=0)
#         fxx_mean = fx_mean.clone()
#         fx_mean[fx_mean!=0]=1
#         ls = torch.sum(abs(fx_mean*pred-fxx_mean)**2)/len(x)
#         return ls   
         
# class hr_loss(nn.Module):
#     def __init__(self):
#         super(hr_loss,self).__init__()
#         self.name = 'hr_loss'
# #         self.js = 0
# #         self.ls_sum = 0

# # #     def forward(self,fx_mean,pred):
# # # #         pred = nn.functional.interpolate(pred, mx.shape[-2:], mode='bilinear', align_corners=True).to(pred.device)
# # #         x,_,_,_ = torch.where(fx_mean!=0)
# # #         fxx_mean = fx_mean.clone()
# # #         fx_mean[fx_mean!=0]=1
# # #         if len(x)!=0:
# # #             ls = torch.sum(abs(fx_mean*pred-fxx_mean)**2)/len(x)
# # #         else:
# # #             ls = 0
# # #         return ls   
#     def forward(self,mx_single,pred):
# #         pred = nn.functional.interpolate(pred, mx.shape[-2:], mode='bilinear', align_corners=True).to(pred.device)
#         pred_fr = mx_single*pred
#         m_mean = 0
#         y_mean = []
#         mx_c = mx_single.clone()
#         for i in range(mx_single.shape[0]):
#             y_mean = torch.sum(pred_fr[i])/torch.sum(mx_single[i])
#             mx_c[i] = mx_c[i]*y_mean
#         m_mean = abs(pred_fr-mx_c)/torch.sum(mx_single[i])
        

#         return m_mean
class hr_loss(nn.Module): 
    def __init__(self): 
        super(hr_loss, self).__init__() 
        self.name = 'hr_loss' 

    def forward(self, mx_single, pred): 
        pred_fr = mx_single * pred  # 遮罩下的预测值
        loss = 0.0
        # print(mx_single.shape)  # 打印输入掩码形状，调试用

        for i in range(mx_single.shape[0]):  # 对 batch 中每个样本遍历
            if torch.sum(mx_single[i]) == 0:
                continue
            y_mean = torch.sum(pred_fr[i]) / torch.sum(mx_single[i])  # 计算该mask区域的平均预测值
            mx_c_i = mx_single[i] * y_mean  # 构造一个均值张量
            diff = torch.abs(pred_fr[i] - mx_c_i)  # 计算偏差
            loss += torch.sum(diff) / torch.sum(mx_single[i])  # 累加每个样本的loss

        return loss / mx_single.shape[0]  # 返回 batch 的平均 los



def hz_depth_px_monitor(mx_single, pred, eps=1e-4):
    """深度当量层位监控(诊断量, 不参与训练): 对每条层位,
    |tau - 批内均值| / (|dtau/dz| + eps) 的均值, 单位=深度采样。
    分子分母同源缩放会抵消, 免疫"梯度压扁"造成的 tau 单位假收敛。"""
    with torch.no_grad():
        gz = torch.zeros_like(pred)
        gz[..., 1:-1, :] = (pred[..., 2:, :] - pred[..., :-2, :]) / 2.0
        vals = []
        for i in range(mx_single.shape[0]):
            m = mx_single[i] > 0
            if m.sum() == 0:
                continue
            tau = pred.expand_as(mx_single[i])[m]
            g = gz.expand_as(mx_single[i])[m].abs()
            vals.append(((tau - tau.mean()).abs() / (g + eps)).mean())
        return torch.stack(vals).mean().item() if vals else float('nan')


class STRUCTURELossv2(nn.Module):
    def __init__(self,u1,u2,u3,use_ep,ep=None):
        super().__init__()
        self.u = torch.stack([u1,u2,u3])
        self.cos = torch.nn.CosineSimilarity(0,eps=1e-16)
        self.use_ep = use_ep
        self.ep = ep
    def forward(self,z):
    #利用 torch.gradient
        vp1,vp2,vp3 = torch.gradient(z)
        vp = torch.stack([vp1,vp2,vp3])
        if self.use_ep:
            # print(torch.abs(self.cos(vp,self.u)).shape)
            # print(self.ep**8.shape)
            cosx = torch.abs(self.cos(vp,self.u))*(self.ep**8)
        else:
            cosx = (self.cos(vp,self.u))
        return  1-torch.mean(cosx)


class STRUCTURELossu_rgt(nn.Module):
    def __init__(self,u1):
        super().__init__()
        self.u = u1
        self.cos = torch.nn.CosineSimilarity(0,eps=1e-6)
    def forward(self,r_g):
    #利用 torch.gradient
        vp= r_g
        
        cosx = torch.abs(self.cos(vp,self.u))
        return  1-torch.mean(cosx)


class ceb_loss(nn.Module):
    def __init__(self):
        super(ceb_loss,self).__init__()
        self.name = 'ceb'
#         self.js = 0
#         self.ls_sum = 0

    def forward(self,y_pred,y_true):
#         pred = nn.functional.interpolate(pred, mx.shape[-2:], mode='bilinear', align_corners=True).to(pred.device)
        _epsilon = torch.finfo(torch.float32).eps
        y_pred   = torch.clip(y_pred, _epsilon, 1 - _epsilon)
        y_pred   = torch.log(y_pred/ (1 - y_pred))
        count_neg = torch.sum(1. - y_true)
        count_pos = torch.sum(y_true)
        beta = count_neg / (count_neg + count_pos)

        pos_weight = beta / (1 - beta)
        pos_weight= pos_weight.detach()/pos_weight.data
        ceb_loss = F.binary_cross_entropy_with_logits(input=y_pred, target=y_true,pos_weight=pos_weight)
#         ceb_loss = torch.mean(ceb_loss * (1 - beta))

#         return torch.where(torch.equal(count_pos, 0.0), 0.0, ceb_loss)
        return ceb_loss

    
class CB_loss(nn.Module):
    def __init__(self,beta,gamma,epsilon=0.1):
        super(CB_loss, self).__init__()
        self.beta = beta
        self.gamma = gamma
        self.epsilon = epsilon
    def forward(self,logits, labels,loss_type = 'sigmoid'):
        """Compute the Class Balanced Loss between `logits` and the ground truth `labels`.
        Class Balanced Loss: ((1-beta)/(1-beta^n))*Loss(labels, logits)
        where Loss is one of the standard losses used for Neural Networks.
        Args:
          labels: A int tensor of size [batch].
          logits: A float tensor of size [batch, no_of_classes].
          samples_per_cls: A python list of size [no_of_classes].
          no_of_classes: total number of classes. int
          loss_type: string. One of "sigmoid", "focal", "softmax".
          beta: float. Hyperparameter for Class balanced loss.
          gamma: float. Hyperparameter for Focal loss.
        Returns:
          cb_loss: A float tensor representing class balanced loss
        """
        # self.epsilon = 0.1 #labelsmooth
        beta = self.beta
        gamma = self.gamma

        no_of_classes = logits.shape[1]
        samples_per_cls = torch.Tensor([sum(labels == i) for i in range(logits.shape[1])])
        if torch.cuda.is_available():
            samples_per_cls = samples_per_cls.cuda()

        effective_num = 1.0 - torch.pow(beta, samples_per_cls)
        weights = (1.0 - beta) / ((effective_num)+1e-8)
        # print(weights)
        weights = weights / torch.sum(weights) * no_of_classes
        labels =labels.reshape(-1,1)

        labels_one_hot  = torch.zeros(len(labels), no_of_classes).scatter_(1, labels, 1)

        weights = torch.tensor(weights).float()
        if torch.cuda.is_available():
            weights = weights.cuda()
            labels_one_hot = torch.zeros(len(labels), no_of_classes).cuda().scatter_(1, labels, 1).cuda()

        labels_one_hot = (1 - self.epsilon) * labels_one_hot + self.epsilon / no_of_classes
        weights = weights.unsqueeze(0)
        weights = weights.repeat(labels_one_hot.shape[0],1) * labels_one_hot
        weights = weights.sum(1)
        weights = weights.unsqueeze(1)
        weights = weights.repeat(1,no_of_classes)

        if loss_type == "focal":
            cb_loss = focal_loss(labels_one_hot, logits, weights, gamma)
        elif loss_type == "sigmoid":
            cb_loss = F.binary_cross_entropy_with_logits(input = logits,target = labels_one_hot, pos_weight = weights)
        elif loss_type == "softmax":
            pred = logits.softmax(dim = 1)
            cb_loss = F.binary_cross_entropy(input = pred, target = labels_one_hot, weight = weights)
        return cb_loss
    
    
class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1e-5):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice
# 定义数据集

class build_dataset_rgt(Dataset):
    def __init__(self, samples_list, dataset_path, mode, possible_num_hrzs, hrz_grp, bit, sample_rate,
                 input_attr_list=["data"], output_attr_list=["label"], mask=False,max_dp = 1,sigma = 2):
        self.samples_list = samples_list
        self.dataset_path = dataset_path
        self.input_attr_list = input_attr_list
        self.output_attr_list = output_attr_list
        self.mask = mask
        self.mode = mode
        self.max = max_dp
        self.possible_num_hrzs = possible_num_hrzs
        self.hrz_grp = hrz_grp
        self.bit = bit
        self.sample_rate = sample_rate
        self.sigma = sigma
    def __len__(self):
        return len(self.samples_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        sample_file = self.samples_list[idx]
        sample_file_path = os.path.join(self.dataset_path, sample_file)
        sample_dict = np.load(sample_file_path, allow_pickle=True).item()
        
        sample_output = {}
        
        if self.mode in ['Train', 'Valid']:
            sx, ux, fl = sample_dict['seis'], sample_dict['rgt'], sample_dict['fault']
            fx, rx,mx = get_train_sample_from_rgt(ux, self.possible_num_hrzs, self.hrz_grp, self.bit, self.sample_rate, fl=fl) 
            sx = mea_std_norm(sx)
            rgt_init,fxrm,fx_sp,fx_mean,mx_single = get_rgt_init_from_rx(fx,mx,self.sample_rate)
            sample_output['rgt'] = rx[np.newaxis,:,:].astype(np.single)*self.max
            sample_output['frame'] = fxrm[np.newaxis,:,:].astype(np.single)*self.max
            sample_output['frame_sp'] = fx_sp[np.newaxis,:,:].astype(np.single)*self.max
            sample_output['frame_mean'] = fx_mean[np.newaxis,:,:].astype(np.single)*self.max
#             sample_output['rgt_init'] = rgt_init[np.newaxis,:,:].astype(np.single)*10
#             sample_output['gradient'] = sample_dict['gradient'][np.newaxis,:,:].astype(np.single)*10
            sample_output['mx'] = mx_single[np.newaxis,:,:,:].astype(np.single)
#             mx_single_sum = np.sum(mx_single,axis=0)
#             b = np.zeros(mx_single_sum.shape,dtype =np.single)
# #             print(f'mx_single_sum={mx_single_sum.shape}')
#             for i in range(mx_single_sum.shape[1]):
#                 b[:,i] = gaussian_filter(mx_single_sum[:,i], sigma=self.sigma)
#                 b[:,i] = min_max_norm(b[:,i])
#             sample_output['mask_seis'] = b[np.newaxis,:,:].astype(np.single)
            sample_output['seis'] = sx[np.newaxis,:,:].astype(np.single)
#             sample_output['seis_m'] = ((1-b)*sx)[np.newaxis,:,:].astype(np.single)
            sample_output['fault'] = fl[np.newaxis,:,:].astype(np.single)
            fl_p = fl.copy()
            max_f = int(np.max(fl_p).item())  # 获取最大断层编号
            if max_f > 0:  # 确保至少有一个断层
                fl_num = random.randint(1, max_f)  # 生成[1, max_f]间的一个随机数
                fl_idx = random.sample(range(1, max_f + 1), fl_num)  # 从[1, max_f]中随机选择fl_num个不同的断层编号
                for j in range(1, max_f + 1):
                    if j in fl_idx:
                        fl_p[fl_p == j] = 1  # 如果j在选择的断层编号中，将相应位置设为1
                    else:
                        fl_p[fl_p == j] = 0  # 否则，将相应位置设为0      
            sample_output['fault_p'] = fl_p[np.newaxis,:,:].astype(np.single)
            
        elif self.mode == 'Infer':
            sx, ux, fl = sample_dict['seis'], sample_dict['rgt'], sample_dict['fault']
            fx, _ = get_train_sample_from_rgt(ux, self.possible_num_hrzs, self.hrz_grp, fl=fl) 
            sx = min_max_norm(sx)
            sample_output['frame'] = fx[np.newaxis,:,:].astype(np.single)
            sample_output['seis'] = sx[np.newaxis,:,:].astype(np.single)
            sample_output['fault'] = fl[np.newaxis,:,:].astype(np.single)
         

        sample_output["mask"] = sample_output['frame'].astype(np.bool_).astype(np.single)
#         mx_len = np.zeros((5,mx.shape[0]),dtype =np.single)
#         for j in range(5):
#             for i in range(mx.shape[0]):
#                 mx_rs = mx_single[:,::2**(4-j),::2**(4-j)]

#                 x,y = np.where(mx_rs[i] != 0)
#                 if len(x)>0:
#                     mx_len[j,i] = len(x)
#                 else:
#                     mx_len[j,i] = 1
#         sample_output['mx_len'] = mx_len[np.newaxis,:,:].astype(np.single)
    
        return  sample_output  


def pred_dict_2d23d_rgt_fl(model, samples,values=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    pred_samples = []

    with torch.no_grad(): 
        for i, sample_pred in enumerate(samples):     

            data = mea_std_norm(sample_pred["seis"])
            data = torch.from_numpy(data).unsqueeze(0).float()
            frame = (sample_pred["frame"])
            frame = torch.from_numpy(frame).unsqueeze(0).float()        
    
                        
            sample_pred['mask'] = sample_pred['frame'].astype(np.bool_).astype(np.single)
            mask = torch.from_numpy(sample_pred['mask']).unsqueeze(0).float()
            
            data, frame, mask = data.to(device), frame.to(device), mask.to(device)
            data, frame, mask = Variable(data), Variable(frame), Variable(mask)
#             data = torch.cat((data, data,data), dim=1)
            data = torch.cat((frame*10, data,data), dim=1)

            target_hr,target_fl= model(data,800) 
            target_fl = torch.sigmoid(target_fl)

            target_hr = target_hr.cpu().squeeze(0).numpy()   
            target_fl = target_fl.cpu().squeeze(0).numpy()
            
            sample_pred["pred"] = target_hr/10
            sample_pred["pred_fl"] = (target_fl)
            sample_pred["frame"] =  sample_pred['fr_mean']



            pred_samples.append(sample_pred)
            
    return pred_samples


class build_dataset_rgt_3d(Dataset):
    def __init__(self, samples_list, dataset_path, mode, possible_num_hrzs, hrz_grp, bit, sample_rate,
                 input_attr_list=["data"], output_attr_list=["label"], mask=False, max_dp=1, sigma=2,
                 crop=False, crop_shape=(64, 128, 128)):
        self.samples_list = samples_list
        self.dataset_path = dataset_path
        self.input_attr_list = input_attr_list
        self.output_attr_list = output_attr_list
        self.mask = mask
        self.mode = mode
        self.max = max_dp
        self.possible_num_hrzs = possible_num_hrzs
        self.hrz_grp = hrz_grp
        self.bit = bit
        self.sample_rate = sample_rate
        self.sigma = sigma
        self.crop = crop
        self.crop_shape = crop_shape  # (D, H, W)

    def __len__(self):
        return len(self.samples_list)

    def crop_data_3d_with_start(self, data, start_d, start_h, start_w):
        cd, ch, cw = self.crop_shape
        return data[start_d:start_d+cd, start_h:start_h+ch, start_w:start_w+cw]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        sample_file = self.samples_list[idx]
        sample_file_path = os.path.join(self.dataset_path, sample_file)
        sample_dict = np.load(sample_file_path, allow_pickle=True).item()
        
        sample_output = {}

        # 读取三维数据
        sx, ux, fl,segments = sample_dict['seis'], sample_dict['rgt'], sample_dict['fault'], sample_dict['segments']
        if self.crop:
            d, h, w = sx.shape
            cd, ch, cw = self.crop_shape
            sd = random.randint(0, d - cd)
            sh = random.randint(0, h - ch)
            sw = random.randint(0, w - cw)
            
            sx = self.crop_data_3d_with_start(sx, sd, sh, sw)
            ux = self.crop_data_3d_with_start(ux, sd, sh, sw)
            fl = self.crop_data_3d_with_start(fl, sd, sh, sw)
            segments = self.crop_data_3d_with_start(segments, sd, sh, sw)
        # 对裁剪后的rgt做归一化
        sx = mea_std_norm(sx)
        ux = min_max_norm(ux)

        if self.mode in ['Train', 'Valid']:
            # fx, rx, mx = get_train_sample_from_rgt_3d(ux, self.possible_num_hrzs, self.hrz_grp, self.bit, self.sample_rate, fl=fl)
            sx = mea_std_norm(sx)
            # rgt_init, fxrm, fx_sp, fx_mean, mx_single = get_rgt_init_from_rx_3d(fx, mx, self.sample_rate)
            sample_output['rgt'] = ux[np.newaxis, ...].astype(np.single) * self.max
            # sample_output['frame'] = fxrm[np.newaxis, ...].astype(np.single) * self.max
            # sample_output['frame_sp'] = fx_sp[np.newaxis, ...].astype(np.single) * self.max
            # sample_output['frame_mean'] = fx_mean[np.newaxis, ...].astype(np.single) * self.max
            sample_output['segments'] = segments[np.newaxis, ...].astype(np.single)
            # 对 mx_single 做 padding
            # max_hrzs = max(self.possible_num_hrzs)
            # pad_shape = (1, max_hrzs) + mx_single.shape[1:]  # [1, max_hrzs, D, H, W]
            # mx_single_pad = np.zeros(pad_shape, dtype=np.single)
            # n_hrzs = mx_single.shape[0]
            # mx_single_pad[0, :n_hrzs, ...] = mx_single[:n_hrzs, ...]
            # sample_output['mx'] = mx_single_pad

            sample_output['seis'] = sx[np.newaxis, ...].astype(np.single)
            sample_output['fault'] = fl[np.newaxis, ...].astype(np.single)
            # 断层处理与2D类似
            # fl_p = fl.copy()
            # max_f = int(np.max(fl_p).item())
            # if max_f > 0:
            #     fl_num = random.randint(1, max_f)
            #     fl_idx = random.sample(range(1, max_f + 1), fl_num)
            #     for j in range(1, max_f + 1):
            #         if j in fl_idx:
            #             fl_p[fl_p == j] = 1
            #         else:
            #             fl_p[fl_p == j] = 0
            # sample_output['fault_p'] = fl_p[np.newaxis, ...].astype(np.single)
        elif self.mode == 'Infer':
            fx, _ = get_train_sample_from_rgt_3d(ux, self.possible_num_hrzs, self.hrz_grp, self.bit, self.sample_rate, fl=fl)
            sx = min_max_norm(sx)
            sample_output['frame'] = fx[np.newaxis, ...].astype(np.single)
            sample_output['seis'] = sx[np.newaxis, ...].astype(np.single)
            sample_output['fault'] = fl[np.newaxis, ...].astype(np.single)

        # sample_output["mask"] = sample_output['frame'].astype(np.bool_).astype(np.single)

        # 在 return sample_output 之前加
        # for k, v in sample_output.items():
        #     if k == "mx":
        #         if isinstance(v, np.ndarray):
        #             print(f"[{self.mode}] idx={idx} key={k} shape={v.shape} dtype={v.dtype}")
        #         else:
        #             print(f"[{self.mode}] idx={idx} key={k} type={type(v)}")
        return sample_output

def get_rgt_init_from_rx(fx,mx,sample_width):
    
    f1 = np.zeros(mx.shape)
    f2 = np.zeros(mx.shape,dtype = np.single)
    mx_single = np.zeros(mx.shape,dtype = np.single)
#     mx_single_all = np.zeros(mx.shape[1:],dtype = np.single)
    fxrm = np.zeros(mx.shape[1:])
    rgtrm = np.zeros(mx.shape[1:])
    rgt_hr = np.zeros(sample_width)
    fr_mean = np.zeros(mx.shape[1:])
    for i in range(mx.shape[0]):

        f1[i] = mx[i,]*fx

        for j in range(mx.shape[2]-1):
            for k in range(mx.shape[1]-1):
                if f1[i,k-1,j] <0.0000001 and f1[i,k,j] >0.00001:
                    f2[i,k,j] = f1[i,k,j]
                    mx_single[i,k,j] =1
#                     mx_single_all[k,j] = 1
                    break
        
        x_all,y_all =np.where(f2[i,:,:] !=0)
        
        if len(x_all) ==0:
            continue
        sum_fr = np.sum(f2[i,:,:])/len(x_all)
        fr_mean += np.int64(f2[i,:,:]!=0)*sum_fr
        for l in range(sample_width):
            rgt_hr[l] = ((np.sum(x_all)+len(x_all)*(l-int(sample_width/2)))/(len(x_all)*(mx.shape[1]-1)))

        for k in range(len(x_all)):
            for j in range(sample_width):
                fxrm[x_all[k]-int(sample_width/2)+j,y_all[k]] = rgt_hr[j]
    fr_sp = f2.sum(axis=(0))
    x,y = np.where(fxrm!=0)
    y_r = sorted(np.unique(y))
    range_x=range(mx.shape[1])
    for i in range(len(y_r)):
        x_r = np.where(fxrm[:,y_r[i]]!=0)

        f = interp1d(x_r[0], fxrm[x_r[0],y_r[i]], fill_value='extrapolate')
#         for j in range(128):
        rgtrm[range_x,y_r[i]] = f(range_x)

    return rgtrm,fxrm,fr_sp,fr_mean,mx_single


def get_train_sample_from_rgt_3d(rx, possible_num_hrzs, hrz_grp, bit=256, sample_rate=2, fl=None):
    """
    适用于三维数据的训练样本生成函数
    rx: 3D numpy array
    fl: 3D numpy array, fault mask
    返回: fx, rx, mx
    """
    d, h, w = rx.shape
    if fl is None:
        fl = np.zeros(rx.shape)
    num_hrzs = random.choice(possible_num_hrzs)
    num_valid_hrzs = bit

    ux = min_max_norm(rx)
    ux = ux * (num_valid_hrzs-1)

    valid_hrzs_idxs = sorted(np.unique(np.around(ux)).tolist())[12:-12]
    valid_hrzs_idxs = [d for i, d in enumerate(valid_hrzs_idxs) if i % sample_rate == 0]

    itv_js = int(len(valid_hrzs_idxs) / num_hrzs)
    hrzs_idxs = []
    for j in range(num_hrzs-1):
        hrzs_idxs += random.sample(valid_hrzs_idxs[j*itv_js:(j+1)*itv_js], 1)
    hrzs_idxs += random.sample(valid_hrzs_idxs[(num_hrzs-1)*itv_js:], 1)

    fx = np.zeros(ux.shape)
    mx = np.zeros((len(hrzs_idxs), d, h, w), dtype=np.single)

    # print("valid_hrzs_idxs:", valid_hrzs_idxs)
    # print("hrzs_idxs:", hrzs_idxs)
    for k, hrzs_idx in enumerate(hrzs_idxs):
        x, y, z = np.where((ux >= hrzs_idx - sample_rate/2) & (ux < (hrzs_idx + sample_rate/2)))
        # print(f"hrzs_idx={hrzs_idx}, num_points={len(x)}")
        for i in range(len(x)):
            # 可根据 fl 做断层mask处理
            if fl[x[i], y[i], z[i]] > 0:
                continue
            fx[x[i], y[i], z[i]] = ux[x[i], y[i], z[i]]
            mx[k, x[i], y[i], z[i]] = 1
    return fx/(bit-1), ux/(bit-1), mx

def get_rgt_init_from_rx_3d(fx, mx, sample_width):
    """
    适用于三维数据的rgt初始化，返回 mx_single
    fx: 3D numpy array
    mx: 4D numpy array (N, D, H, W)
    sample_width: int
    返回: rgtrm, fxrm, fr_sp, fr_mean, mx_single
    """
    f1 = np.zeros(mx.shape, dtype=np.single)
    f2 = np.zeros(mx.shape, dtype=np.single)
    mx_single = np.zeros(mx.shape, dtype=np.single)
    fxrm = np.zeros(mx.shape[1:], dtype=np.single)
    rgtrm = np.zeros(mx.shape[1:], dtype=np.single)
    rgt_hr = np.zeros(sample_width)
    fr_mean = np.zeros(mx.shape[1:], dtype=np.single)
    for i in range(mx.shape[0]):
        f1[i] = mx[i] * fx
        for l in range(mx.shape[3]):
            for j in range(mx.shape[2]):
                for k in range(1, mx.shape[1]):
                    if f1[i, k-1, j, l] < 1e-6 and f1[i, k, j, l] > 1e-6:
                        f2[i, k, j, l] = f1[i, k, j, l]
                        mx_single[i, k, j, l] = 1
                        break
        x_all, y_all, z_all = np.where(f2[i] != 0)
        if len(x_all) == 0:
            continue
        sum_fr = np.sum(f2[i]) / len(x_all)
        fr_mean += (f2[i] != 0) * sum_fr
        for l in range(sample_width):
            rgt_hr[l] = ((np.sum(x_all) + len(x_all) * (l - 1)) / (len(x_all) * (mx.shape[1] - 1)))
        for k in range(len(x_all)):
            for j in range(sample_width):
                xi = x_all[k] - 1 + j
                if 0 <= xi < fxrm.shape[0]:
                    fxrm[xi, y_all[k], z_all[k]] = rgt_hr[j]
    fr_sp = f2.sum(axis=0)
    x, y, z = np.where(fxrm != 0)
    y_r = sorted(np.unique(y))
    z_r = sorted(np.unique(z))
    range_x = range(mx.shape[1])
    for j in z_r:
        for i in y_r:
            x_r = np.where(fxrm[:, i, j] != 0)[0]
            if len(x_r) > 1:
                f = interp1d(x_r, fxrm[x_r, i, j], fill_value='extrapolate')
                rgtrm[:, i, j] = f(range_x)
            elif len(x_r) == 1:
                rgtrm[:, i, j] = fxrm[x_r[0], i, j]
    return rgtrm, fxrm, fr_sp, fr_mean, mx_single

class build_dataset_cigfacies(Dataset):
    def __init__(self, samples_list, dataset_path, mode, frame_part = False,mx_valid = False,
                 input_attr_list=["seis"],input_attr_list2=["cigfacies"],
                 input_attr_list3=["normal"],input_attr_list4=["linearity"],
                 input_attr_list5=["mx_single"],input_attr_list6=["frame"],
                 index_attr_list = ["index"],
                 output_attr_list=["rgt"],output_attr_list2=["unconformities"],norm=None,seg_mode = "2d"):
        self.samples_list = samples_list
        self.dataset_path = dataset_path
        self.input_attr_list = input_attr_list
        self.input_attr_list2 = input_attr_list2
        self.input_attr_list3 = input_attr_list3
        self.input_attr_list4 = input_attr_list4
        self.input_attr_list5 = input_attr_list5
        self.input_attr_list6 = input_attr_list6
        self.output_attr_list = output_attr_list
        self.output_attr_list2 = output_attr_list2
        self.index_attr_list = index_attr_list
        self.seg_mode = seg_mode
        if self.seg_mode == "3d":
            self.input_attr_list_all = [self.input_attr_list,self.input_attr_list5,self.input_attr_list6,["segments"]]     
        elif self.seg_mode == "2d":
            self.input_attr_list_all = [self.input_attr_list,self.input_attr_list2,self.input_attr_list3,
                                        self.input_attr_list4,self.input_attr_list5,self.input_attr_list6]                        
        if mx_valid:
            self.input_attr_list_all.append(['mask_valid'])    
        if frame_part:
            self.input_attr_list_all.append(['frame_part'])
        self.mode = mode
        
    def __len__(self):
        return len(self.samples_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        sample_file_name = self.samples_list[idx]
        sample_file_path = os.path.join(self.dataset_path, sample_file_name + ".npy")
        sample_dict = np.load(sample_file_path, allow_pickle=True).item()
        
        sample_output = {}
        
        if self.mode in ['Train', 'Valid']:
            for i, input_attr in enumerate(self.input_attr_list_all):
                # print(input_attr)
                tmp = sample_dict[input_attr[0]].astype(np.single)
                # tmp = tmp[np.newaxis,:,:]
                # tmp = (min_max_norm(tmp) * 10)
                sample_output[input_attr[0]] = tmp # rgt
            if 'segments_teacher' in sample_dict:
                sample_output['segments_teacher'] = sample_dict['segments_teacher'].astype(np.single)
            if 'segments_w' in sample_dict:
                sample_output['segments_w'] = sample_dict['segments_w'].astype(np.single)
            if 'fault_pairs' in sample_dict:
                sample_output['fault_pairs'] = sample_dict['fault_pairs'].astype(np.single)
            # outputs
            # for i, output_attr in enumerate(self.output_attr_list):
            #     tmp = sample_dict[output_attr].astype(np.single)
            #     tmp = tmp[np.newaxis,:,:]
            #     tmp = (min_max_norm(tmp) * 10)
            #     sample_output[output_attr] = tmp # rgt

            for i, index_attr in enumerate(self.index_attr_list):
                tmp = np.array(int(sample_file_name)).astype(np.int64)
                sample_output[index_attr] = tmp # index data

        sample_output["sample_file_path"] = sample_file_path
        return  sample_output
   
# 计算 position embedding
def pos_embedding(pos_tensor, v_dim, reserve_bit=2):
    
    pos_tensor = np.round(rgt*(10**reserve_bit))
    
    p = pos_tensor.astype(np.float32)
    p = np.expand_dims(p, axis=-1)
    p = p.repeat(v_dim/2, axis=-1)
    
    w = 1. / np.power(10000., 2. * np.arange(v_dim/2. ,dtype=np.float32) / v_dim)
    w = w.astype(np.float32)
    
    wp = w*p
    s = np.sin(wp)
    c = np.cos(wp)
    
    pos_embed = np.concatenate([s,c], axis=-1)
    return pos_embed

# 读取数据体
def read_cube(data_path, data_file, num_inline, num_crossline):
    data_file = os.path.join(data_path, data_file+".dat")
    print(data_file)
    if os.path.exists(data_file):
        cube = np.load(data_file)
        cube = np.reshape(cube, (num_crossline, num_inline, -1))
        return cube
    else:
        return None

# 曲线光滑函数
def smooth(v, w=0.85):
    last = v[0]
    smoothed = []
    for point in v:
        smoothed_val = last * w + (1 - w) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

# 训练和验证






class WarmupCosineAnnealingLR(_LRScheduler):
    def __init__(self, optimizer, total_epochs, warmup_epochs, lr_min=1e-6, last_epoch=-1):
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.lr_min = lr_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        # 当前 epoch
        current_epoch = self.last_epoch + 1
        lrs = []

        for base_lr in self.base_lrs:
            if current_epoch < self.warmup_epochs:
                # warm-up 阶段：线性上升
                lr = base_lr * current_epoch / self.warmup_epochs
            else:
                # cosine annealing 阶段
                progress = (current_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
                cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
                lr = self.lr_min + (base_lr - self.lr_min) * cosine_decay
            lrs.append(lr)
        return lrs


def train_valid_net(param, model, train_data, valid_data, criterion=None,criterion_fl=None, input_attrs=["data"], output_attrs=["label"],
                    plot=True,mtl = False,transfer=None,pr_de = False,facies=True):
    
    #初始化参数
    epochs = param['epochs']
    warm_up = param['warm_up']
    warm_up_epochs = param['warm_up_epochs']
    batch_size = param['batch_size']
    lr = param['lr']
    lr_patience = param['lr_patience']
    lr_factor = param['lr_factor']
    optimizer_type = param['optimizer_type']
    gamma = param['gamma']
    step_size = param['step_size']
    momentum = param['momentum']
    weight_decay = param['weight_decay']
    disp_inter = param['disp_inter']
    save_inter = param['save_inter']
    checkpoint_path = param['checkpoint_path']
    ol_seis = param['ol_seis']
    use_mse = param['use_mse']


    criterion_hr_global = hr_loss()
    criterion_mse = MSELoss()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, drop_last=True,num_workers = 20)
    valid_loader = DataLoader(dataset=valid_data, batch_size=1, shuffle=False,num_workers = 10)
    
    if optimizer_type == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif optimizer_type == "Adam":
#         if transfer:
#             optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
#         else:
#             optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
#         scheduler = ReduceLROnPlateau(optimizer, 'min', patience=lr_patience, factor=lr_factor)
        print('use adam')
        if transfer:
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr,weight_decay=weight_decay)
        else:
            optimizer = optim.Adam(model.parameters(), lr=lr,weight_decay=weight_decay)

        scheduler = MultiStepLR(optimizer, milestones=[100,200,300,400,500,600,700,800], gamma=1) 
    elif optimizer_type == "Adamw":
        print('use adamw')
        if transfer:
            optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr,weight_decay=weight_decay)
        else:
            optimizer = optim.AdamW(model.parameters(), lr=lr,weight_decay=weight_decay)
        if warm_up:
            scheduler = WarmupCosineAnnealingLR(
            optimizer,
            total_epochs=epochs,
            warmup_epochs=warm_up_epochs,
            lr_min=1e-5  # 可设为 base_lr 的 1% 左右
            )
        else:
            scheduler = MultiStepLR(optimizer, milestones=[100,200,300,400,500,600,700,800], gamma=0.5)    
    elif optimizer_type == "Adam_sam":
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5) #learning rate decay

#         scheduler = MultiStepLR(optimizer, milestones=[50,100,150,200,250,300,350,400], gamma=0.5)         
    if criterion is None:
        criterion = nn.MSELoss().to(device)
    
#     criterion_mse = nn.MSELoss().to(device)
#     if criterion_fl is None:
#         criterion_fl = nn.MSELoss().to(device)  
    if facies:
        criterion_facies = SegmentLoss(loss_type="L2").to(device)
    criterion_ce = nn.BCEWithLogitsLoss().to(device)
    criterion_bce = ceb_loss().to(device)

#     beta = 0.9999
#     gamma = 2.0
#     criterion_bce = CB_loss(beta, gamma).to(device)
    print('fault_loss use cbe')

    # 主循环
    epoch_loss_train, epoch_loss_valid, epoch_lr = [], [], []
    epoch_loss_fl_train,epoch_loss_fl_valid,epoch_loss_hr_train,epoch_loss_hr_valid = [],[],[],[]
    epoch_loss_mse_train,epoch_loss_mse_valid = [],[]
    criterion_dice = DiceLoss()
    best_mse = 1e50

    w_chamfer = 0.1
    time_loss_ssim, time_loss_mse, time_loss_facies = 0, 0, 0
    for epoch in range(epochs):
        # since = time.time()
        since = time.time()
        
        # 重置每个epoch的计时
        time_loss_ssim, time_loss_mse, time_loss_facies = 0, 0, 0   
        # 训练阶段
        model.train()
        loss_train_per_epoch = 0
        loss_train_ssim_per_epoch = 0
        loss_train_mse_per_epoch = 0
        loss_train_facies_per_epoch = 0
        loss_train_hr_global_per_epoch = 0
        loss_fl_train_per_epoch = 0
        loss_hr_train_per_epoch = 0
#         loss_train_mse_per_epoch = 0
        
        for batch_idx, batch_samples in enumerate(train_loader):


            data = batch_samples["seis"]
#             data = batch_samples["seis_m"]
#             print('datashape='+str(data.shape))

            target_flgt = batch_samples["fault"]
            for i, output_attr in enumerate(output_attrs):
                tmp = batch_samples[output_attr]
                if i  == 0:
                    target = tmp
                else:
                    target = torch.cat((target, tmp), dim=1)
            # print(f'target.shape={target.shape}')
            data, target,target_flgt = target.to(device), data.to(device),target_flgt.to(device)
            data, target,target_flgt = Variable(target), Variable(data), Variable(target_flgt)

            # mask = batch_samples["mask"]
            # mask = Variable(mask.to(device)) 
            # frame = batch_samples["frame"]
            # frame = Variable(frame.to(device)) 
            # fl_p = batch_samples["fault_p"]
            # fl_p = Variable(fl_p.to(device))
            segments = batch_samples["segments"]
            segments = Variable(segments.to(device)) 
            # frame_mean = batch_samples["frame_mean"]
            # frame_mean = Variable(frame_mean.to(device)) 
            optimizer.zero_grad()
            if ol_seis:
                data = torch.cat((data,data,data), dim=1)
            else:
                data = torch.cat((target*mask,data,data), dim=1)
            # print(f'target shape={target.shape}')
            # print(f'mask shape={mask.shape}')
            # print(f'target*mask shape={(target*mask).shape}')
#             data = torch.cat((data,data,data), dim=1)
            if mtl == False:
                
#                 target_i = model(data,mx,mx_len)
                if pr_de:
                    target_i = model(data,target*mask,fl_p)
                else:
                    target_i = model(data)
                target_j =  target_i
                # print(f'predi.max() ={target_i.max()}')
                # print(f'target.max() ={target.max()}')
                # print(f'target*mask.max() ={(target*mask).max()}')
                if not ol_seis:
                    l_hr_global = 0
                # print(f'frame_mean.max() ={frame_mean.max()}')
                if ol_seis:
                    target_j = target_j
                else:
                    target_j =  target_j * (1-mask) + target *(mask)
                # print(f'predj.max() ={target_j.max()}')
#                 print(f'input size={target_j.shape}')
#                 print(f'target size={target.shape}')
                # l_ssim = 1 - ms_ssim3d(target_j, target, data_range=1.0)   
                # t0 = time.perf_counter()
                # l_ssim = criterion(target_j, target)
                # l_ssim = 0
                # time_loss_ssim += time.perf_counter() - t0
                
                if ol_seis:
                    if use_mse:
                        t0 = time.perf_counter()
                        l_mse = criterion_mse(target_j, target)
                        loss_train_mse_per_epoch += l_mse.item()
                        time_loss_mse += time.perf_counter() - t0
                        
                        if facies:
                            t0 = time.perf_counter()
                            l_facies = criterion_facies(target_j,segments)
                            loss_train_facies_per_epoch += l_facies.item()
                            time_loss_facies += time.perf_counter() - t0                      
                else:
                    if use_mse:
                        l_mse = criterion_mse(target_j, target)
                        loss_train_mse_per_epoch += l_mse.item()
                        
                        if facies:
                            l_facies = criterion_facies(target_j,segments)
                            loss_train_facies_per_epoch += l_facies.item()
                            loss = l_ssim + l_mse+0.01*l_facies
                    else:
                        loss = l_ssim+l_hr_global
                # loss = l_ssim
                if use_mse:
                    loss = l_mse
                if facies:
                    loss += 0.01 * l_facies
                loss.backward()
                optimizer.step()
                if math.isnan(loss) == True:
                    print('train:'+str(batch_idx)) 
                    print(f'error_loss={loss}')

                loss_train_per_epoch += loss.item()
                # loss_train_ssim_per_epoch += l_ssim.item()
                # if not ol_seis:
                    # loss_train_hr_global_per_epoch += l_hr_global.item()



            elif mtl:
                target_hr,target_fl= model(data,800)

#                 channelPoint = torch.sum(target_flgt)
#                 allPoint = target_flgt.shape[0]*target_flgt.shape[1]*target_flgt.shape[2]*target_flgt.shape[3]
#                 PointRatio = allPoint//channelPoint

#                 weight_CE = torch.FloatTensor([1,PointRatio]).to(device)
#                 print(f"weight_CE={weight_CE.shape}")
#                 criterion_bce = nn.BCEWithLogitsLoss(weight=weight_CE).to(device)


    
   
#                 if epoch % 10 ==0:
   
#                 if epoch <400 :
#                     l_hr_global = criterion_hr_global(frame_mean,target_hr)
#                     target_hr =  target_hr * (1-mask) + target *(mask)
#                     l_ssim = criterion(target_hr, target)
#                     loss_hr = l_ssim+l_hr_global
# #                     loss_fl = criterion_fl((target_fl),target_flgt)
# #                     loss = (10*loss_hr+loss_fl)/2
# #                     loss_fl_train_per_epoch += loss_fl.item()  
#                     loss = loss_hr
#                 elif 600>epoch >400:
#                     loss_fl = criterion_fl((target_flgt),target_fl)
#                     loss = loss_fl
#                     loss_fl_train_per_epoch += loss_fl.item() 
                if epoch>=0:
#                 else:
                    l_hr_global = criterion_hr_global(frame_mean,target_hr)
                    target_hr =  target_hr * (1-mask) + target *(mask)
                    l_ssim = criterion(target_hr, target)
                    loss_hr = l_ssim+l_hr_global
                    loss_fl = 0.05*criterion_bce(target_fl,target_flgt)+0.95*criterion_dice(target_fl,target_flgt)
#                     loss_fl = criterion_ce(target_fl,target_flgt)
                    loss = (10*loss_hr+loss_fl)/2
                    loss_fl_train_per_epoch += loss_fl.item()
                    
                loss.backward()
                optimizer.step()
#                 if math.isnan(loss_hr) == True or math.isnan(loss_fl) == True:
#                     print('train:'+str(batch_idx)) 
                    
                loss_train_per_epoch += loss.item()
#                 loss_train_ssim_per_epoch += l_ssim.item()
#                 loss_train_hr_global_per_epoch += l_hr_global.item()

#                 loss_hr_train_per_epoch += loss_hr.item()     
        # print(f"Epoch {epoch}:")
        # print(f"  SSIM loss time: {time_loss_ssim:.4f} s")
        # print(f"  MSE loss time: {time_loss_mse:.4f} s")
        # print(f"  Facies loss time: {time_loss_facies:.4f} s")
        # print(f"  Epoch total time: {time.time() - since:.2f} s")
        # 验证阶段
        model.eval()
        loss_valid_per_epoch = 0
        loss_valid_ssim_per_epoch = 0
        loss_mse_valid_per_epoch = 0
        loss_facies_valid_per_epoch = 0
        loss_valid_hr_global_per_epoch = 0
        loss_fl_valid_per_epoch = 0
        loss_hr_valid_per_epoch = 0
        loss_valid_mse_per_epoch = 0
        loss_valid_facies_per_epoch = 0
        with torch.no_grad():
            for batch_idx, batch_samples in enumerate(valid_loader):   
                
                data = batch_samples["seis"]
#                 data = batch_samples["seis_m"]
                target_flgt = batch_samples["fault"]
                for i, output_attr in enumerate(output_attrs):
                    tmp = batch_samples[output_attr]
                    if i  == 0:
                        target = tmp
                    else:
                        target = torch.cat((target, tmp), dim=1)

                data, target,target_flgt = target.to(device), data.to(device),target_flgt.to(device)
                data, target,target_flgt = Variable(target), Variable(data), Variable(target_flgt)
                # frame = batch_samples["frame"]
                # frame = Variable(frame.to(device)) 
                # mask = batch_samples["mask"]
                # mask = Variable(mask.to(device)) 
                segments = batch_samples["segments"]
                segments = Variable(segments.to(device)) 
#                 fl_p = batch_samples["fault_p"]
#                 fl_p = Variable(fl_p.to(device))
# #                 mx_len = batch_samples["mx_len"]
# #                 mx_len = Variable(mx_len.to(device))
# #                 mx = batch_samples["mx"]
# #                 mx = Variable(mx.to(device)) 
#                 frame_mean = batch_samples["frame_mean"]
#                 frame_mean = Variable(frame_mean.to(device)) 
                if ol_seis:
                    data = torch.cat((data,data,data), dim=1)
                else:
                    data = torch.cat((target*mask,data,data), dim=1)
#                 data = torch.cat((data,data,data), dim=1)               


                if mtl == False:

#                     target_i = model(data,mx,mx_len)
                    if ol_seis:
                        target_i = model(data)
                    else:
                        target_i = model(data)

                    



                    target_j =  target_i
                    # print(f'predi.max() ={target_i.max()}')
                    # print(f'predj.max() ={target_j.max()}')
                    if not ol_seis:
                        # l_hr_global_valid = criterion_hr_global(torch.squeeze(target*mask,dim=1),target_j)
                        l_hr_global_valid = 0
                    # print(f'l_hr_global_valid.max() ={l_hr_global_valid.max()}')
#                     target_j =  target_j * (1-mask) + target *(mask)              

                    # l_ssim_valid = criterion(target_j, target) 
                    if ol_seis:
                        # loss_valid = l_ssim_valid
                        if use_mse:
                            l_mse_valid = criterion_mse(target_j, target)
                            loss_mse_valid_per_epoch = l_mse_valid.item()
                            if facies:
                                l_facies_valid = criterion_facies(target_j,segments)
                                loss_facies_valid_per_epoch += l_facies_valid.item()
                                # loss_valid = l_ssim_valid + l_mse_valid +0.01*l_facies_valid
                    else:
                        if use_mse:
                            l_mse_valid = criterion_mse(target_j, target)
                            loss_mse_valid_per_epoch += l_mse_valid.item()
                            if facies:
                                l_facies_valid = criterion_facies(target_j,segments)
                                loss_facies_valid_per_epoch += l_facies_valid.item()
                                loss_valid = l_ssim_valid + l_mse_valid + l_hr_global_valid+0.01*l_facies_valid
                        else:
                            loss_valid = l_ssim_valid+l_hr_global_valid
                    # loss_valid = l_ssim_valid
                    if use_mse:
                        loss_valid = l_mse_valid
                    if facies:
                        loss_valid += 0.01 * l_facies_valid
                    loss_valid_per_epoch += loss_valid.item()
                    # loss_valid_ssim_per_epoch += l_ssim_valid.item()
                    # if not ol_seis:
                        # loss_valid_hr_global_per_epoch += l_hr_global_valid.item()
              
                elif mtl:

                    
                    
                    target_hr,target_fl= model(data,800)

#                     l_hr_global_valid = criterion_hr_global(frame_mean,target_hr)
#                     target_hr =  target_hr * (1-mask) + target *(mask)   
#                     l_ssim_valid = criterion(target_hr, target)
#                     loss_hr_valid = l_ssim_valid+l_hr_global_valid
#                     if epoch % 10 ==0:
#                         loss_fl_valid = criterion_fl((target_fl),target_flgt)
#                         loss_valid = (10*loss_hr_valid+loss_fl_valid)/2
#                         loss_fl_valid_per_epoch += loss_fl_valid.item()
#                     else:
#                         loss_valid = loss_hr_valid
                        
                        
#                     if epoch <400 :
#                         l_hr_global_valid = criterion_hr_global(frame_mean,target_hr)
#                         target_hr =  target_hr * (1-mask) + target *(mask)
#                         l_ssim_valid = criterion(target_hr, target)
#                         loss_hr_valid = l_ssim_valid+l_hr_global_valid
#     #                     loss_fl = criterion_fl((target_fl),target_flgt)
#     #                     loss = (10*loss_hr+loss_fl)/2
#     #                     loss_fl_train_per_epoch += loss_fl.item()  
#                         loss_valid = loss_hr_valid
#                     elif 600>epoch >400:
#                         loss_fl_valid = criterion_fl((target_flgt),target_fl)
#                         loss_valid = loss_fl_valid
#                         loss_fl_valid_per_epoch += loss_fl_valid.item()
#                     else:
                    if epoch >=0:
                        l_hr_global_valid = criterion_hr_global(frame_mean,target_hr)
                        target_hr =  target_hr * (1-mask) + target *(mask)
                        l_ssim_valid = criterion(target_hr, target)
                        loss_hr_valid = l_ssim_valid+l_hr_global_valid
                        loss_fl_valid = 0.05*criterion_bce(target_fl,target_flgt)+0.95*criterion_dice(target_fl,target_flgt)
#                         loss_fl_valid = criterion_ce(target_fl,target_flgt)
                        loss_valid = (10*loss_hr_valid+loss_fl_valid)/2
                        loss_fl_valid_per_epoch += loss_fl_valid.item()
                        

                    loss_valid_per_epoch += loss_valid.item()                                     
                    loss_hr_valid_per_epoch += loss_hr_valid.item()  
                    # loss_valid_ssim_per_epoch += l_ssim_valid.item()
                    # loss_valid_hr_global_per_epoch += l_hr_global_valid.item()
                    
        loss_train_per_epoch = loss_train_per_epoch / len(train_loader)
        loss_train_ssim_per_epoch = loss_train_ssim_per_epoch / len(train_loader)
        loss_train_mse_per_epoch = loss_train_mse_per_epoch / len(train_loader)
        loss_train_facies_per_epoch = loss_train_facies_per_epoch / len(train_loader)
        loss_train_hr_global_per_epoch = loss_train_hr_global_per_epoch / len(train_loader)
        loss_valid_per_epoch = loss_valid_per_epoch / len(valid_loader)
        loss_valid_ssim_per_epoch = loss_valid_ssim_per_epoch / len(valid_loader)
        loss_valid_mse_per_epoch = loss_mse_valid_per_epoch / len(valid_loader)
        loss_valid_facies_per_epoch = loss_facies_valid_per_epoch / len(valid_loader)
        loss_valid_hr_global_per_epoch = loss_valid_hr_global_per_epoch / len(valid_loader)
        
        if mtl:
            loss_fl_train_per_epoch = loss_fl_train_per_epoch / len(train_loader)
            loss_fl_valid_per_epoch = loss_fl_valid_per_epoch / len(valid_loader)
            loss_hr_train_per_epoch = loss_hr_train_per_epoch / len(train_loader)
            loss_hr_valid_per_epoch = loss_hr_valid_per_epoch / len(valid_loader)
        epoch_loss_train.append(loss_train_per_epoch)
        epoch_loss_valid.append(loss_valid_per_epoch)
        if mtl:
            epoch_loss_fl_train.append(loss_fl_train_per_epoch)
            epoch_loss_fl_valid.append(loss_fl_valid_per_epoch)
            epoch_loss_hr_train.append(loss_hr_train_per_epoch)
            epoch_loss_hr_valid.append(loss_hr_valid_per_epoch)
        epoch_lr.append(optimizer.param_groups[0]['lr'])

        # 保存模型
        if epoch % save_inter == 0:
            state = {'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
            filename = os.path.join(checkpoint_path, 'checkpoint-epoch{}.pth'.format(epoch))
            torch.save(state, filename)
            # torch.save(lora.lora_state_dict(model), filename)
            # print(filename)

        # 保存最优模型
        if loss_valid_per_epoch < best_mse: # loss_per_epoch valid_mse_per_epoch
            state = {'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
            filename = os.path.join(checkpoint_path, 'checkpoint-best.pth')
            torch.save(state, filename)
            # torch.save(lora.lora_state_dict(model), filename)
            best_mse = loss_valid_per_epoch

#         scheduler.step(loss_train_per_epoch)
        scheduler.step()
        time_elapsed = time.time() - since
#         print('Training complete in {:.0f}m {:.0f}s'.format(
#             time_elapsed // 60, time_elapsed % 60))
        # 显示loss
        if epoch % disp_inter == 0: 
            if mtl:
                print('Epoch:{}, Training Loss:fl={:.8f} hr_all:{:.8f} si:{:.8f} hr:{:.8f} Validation Loss:fl={:.8f} si:{:.8f} hr:{:.8f} Learning rate: {:.8f} time:{:.0f}m {:.0f}s'.format(epoch, loss_fl_train_per_epoch,loss_hr_train_per_epoch,loss_train_ssim_per_epoch,loss_train_hr_global_per_epoch,loss_fl_valid_per_epoch,loss_valid_ssim_per_epoch,loss_valid_hr_global_per_epoch, epoch_lr[epoch],time_elapsed // 60, time_elapsed % 60))
            if not mtl:
                print('Epoch:{}, Training Loss:{:.8f} si:{:.8f} mse:{:.8f} facies:{:.8f} hr:{:.8f} Validation Loss:{:.8f} si:{:.8f} mse:{:.8f} facies:{:.8f} hr:{:.8f}  Learning rate: {:.8f} time:{:.0f}m {:.0f}s'.format(epoch, loss_train_per_epoch,loss_train_ssim_per_epoch,loss_train_mse_per_epoch,loss_train_facies_per_epoch,loss_train_hr_global_per_epoch, loss_valid_per_epoch,loss_valid_ssim_per_epoch,loss_valid_mse_per_epoch,loss_valid_facies_per_epoch,loss_valid_hr_global_per_epoch, epoch_lr[epoch],time_elapsed // 60, time_elapsed % 60))      
#                 print('Epoch:{}, Training Loss:{:.8f} Validation Loss:{:.8f} Learning rate: {:.8f} time:{:.0f}m {:.0f}s'.format(epoch, loss_train_per_epoch, loss_valid_per_epoch, epoch_lr[epoch],time_elapsed // 60, time_elapsed % 60))

    # 训练loss曲线
    if plot:
        if mtl == False:
            x = [i for i in range(epochs)]
            fig = plt.figure(figsize=(12, 4))
            ax = fig.add_subplot(1, 2, 1)
            ax.plot(x, smooth(epoch_loss_train, 0.6), label='Training loss')
            ax.plot(x, smooth(epoch_loss_valid, 0.6), label='Validation loss')
            ax.set_xlabel('Epoch', fontsize=15)
            ax.set_ylabel('Loss', fontsize=15)
            ax.set_title(f'Training curve', fontsize=15)
            ax.grid(True)
            plt.legend(loc='upper right', fontsize=15)


            ax = fig.add_subplot(1, 2, 2)
            ax.plot(x, epoch_lr,  label='Learning Rate')
            ax.set_xlabel('Epoch', fontsize=15)
            ax.set_ylabel('Learning Rate', fontsize=15)
            ax.set_title(f'Learning rate curve', fontsize=15)
            ax.grid(True)
            plt.legend(loc='upper right', fontsize=15)
            plt.show()
        else:
            x = [i for i in range(epochs)]
            fig = plt.figure(figsize=(12, 4))
            ax = fig.add_subplot(2, 2, 1)
            ax.plot(x, smooth(epoch_loss_train, 0.6), label='Training loss')
            ax.plot(x, smooth(epoch_loss_valid, 0.6), label='Validation loss')
            ax.set_xlabel('Epoch', fontsize=15)
            ax.set_ylabel('Loss', fontsize=15)
            ax.set_title(f'Training curve', fontsize=15)
            ax.grid(True)
            plt.legend(loc='upper right', fontsize=15)

            ax = fig.add_subplot(2, 2, 2)
            ax.plot(x, smooth(epoch_loss_fl_train, 0.6), label='Training loss')
            ax.plot(x, smooth(epoch_loss_fl_valid, 0.6), label='Validation loss')
            ax.set_xlabel('Epoch', fontsize=15)
            ax.set_ylabel('Loss', fontsize=15)
            ax.set_title(f'Training fault curve', fontsize=15)
            ax.grid(True)
            plt.legend(loc='upper right', fontsize=15)
            
            ax = fig.add_subplot(2, 2, 3)
            ax.plot(x, smooth(epoch_loss_hr_train, 0.6), label='Training loss')
            ax.plot(x, smooth(epoch_loss_hr_valid, 0.6), label='Validation loss')
            ax.set_xlabel('Epoch', fontsize=15)
            ax.set_ylabel('Loss', fontsize=15)
            ax.set_title(f'Training horizons curve', fontsize=15)
            ax.grid(True)
            plt.legend(loc='upper right', fontsize=15)
            

            ax = fig.add_subplot(2, 2, 4)
            ax.plot(x, epoch_lr,  label='Learning Rate')
            ax.set_xlabel('Epoch', fontsize=15)
            ax.set_ylabel('Learning Rate', fontsize=15)
            ax.set_title(f'Learning rate curve', fontsize=15)
            ax.grid(True)
            plt.legend(loc='upper right', fontsize=15)
            plt.show()            
    if mtl == True:      
        logs = {"epoch_loss_train":epoch_loss_train,
                "epoch_loss_valid":epoch_loss_valid,
                "epoch_loss_fl_train":epoch_loss_fl_train,
                "epoch_loss_fl_valid":epoch_loss_fl_valid,
                "epoch_loss_hr_train":epoch_loss_hr_train,
                "epoch_loss_hr_valid":epoch_loss_hr_valid,
                "epoch_lr":epoch_lr}
    else:
        logs = {"epoch_loss_train":epoch_loss_train,
                "epoch_lr":epoch_lr}
    if qc_history:
        logs["qc_history"] = qc_history
    np.save(os.path.join(checkpoint_path, 'logs.npy'), logs)



    return model



_PHASE_LOSS_SHAPE_WARNED = False

def seismic_phase_alignment_loss(pred_rgt, seismic, eps=1e-6, amp_percentile=70.0,
                                 penalty='l1', penalty_scale=0.3):
    """
    Encourage RGT iso-lines to follow seismic phase events, without teacher RGT.

    In 2D, grad(RGT) is normal to an RGT iso-line. A tangent direction is
    (dRGT/dx, -dRGT/dz). If an iso-line follows a reflector, the directional
    derivative of seismic amplitude along this tangent should be small.

    penalty:
        'l1'          - |d|，原始行为。断层处大残差会被线性惩罚，倾向抹平断距
        'charbonnier' - sqrt(d^2+c^2)-c，平滑 L1
        'welsch'      - 1-exp(-(d/c)^2)，饱和型。残差 >> c 时梯度趋于 0，
                        断层/失配处自动放弃对齐，不再抹平断距
    penalty_scale: c。基于 zxdata ep100 预测实测标定：贴合同相轴的残差
        p50≈0.13 / p90≈0.35，断层尾部 >0.5，故默认 c=0.3（≈p90）
    """
    global _PHASE_LOSS_SHAPE_WARNED
    if (pred_rgt.dim() != 4 or seismic.dim() != 4
            or pred_rgt.shape[-2:] != seismic.shape[-2:]):
        if not _PHASE_LOSS_SHAPE_WARNED:
            print(f"[WARN] seismic_phase_alignment_loss: 形状不匹配 "
                  f"pred={tuple(pred_rgt.shape)} seis={tuple(seismic.shape)}，"
                  f"loss 恒为 0（仅提示一次）")
            _PHASE_LOSS_SHAPE_WARNED = True
        return pred_rgt.sum() * 0.0

    rgt = pred_rgt
    seis = seismic[:, :1, :, :]
    dr_dz = rgt[:, :, 2:, 1:-1] - rgt[:, :, :-2, 1:-1]
    dr_dx = rgt[:, :, 1:-1, 2:] - rgt[:, :, 1:-1, :-2]
    ds_dz = seis[:, :, 2:, 1:-1] - seis[:, :, :-2, 1:-1]
    ds_dx = seis[:, :, 1:-1, 2:] - seis[:, :, 1:-1, :-2]

    tangent_z = dr_dx
    tangent_x = -dr_dz
    tangent_norm = torch.sqrt(tangent_z ** 2 + tangent_x ** 2 + eps)
    directional = (ds_dz * tangent_z + ds_dx * tangent_x) / tangent_norm

    if penalty == 'welsch':
        rho = 1.0 - torch.exp(-(directional / penalty_scale) ** 2)
    elif penalty == 'charbonnier':
        rho = torch.sqrt(directional ** 2 + penalty_scale ** 2) - penalty_scale
    else:
        rho = torch.abs(directional)

    amp = torch.abs(seis[:, :, 1:-1, 1:-1])
    if amp_percentile is not None and amp_percentile > 0:
        flat = amp.detach().reshape(amp.shape[0], -1)
        thr = torch.quantile(flat, amp_percentile / 100.0, dim=1).view(-1, 1, 1, 1)
        weight = (amp >= thr).to(pred_rgt.dtype)
        if torch.sum(weight) > 0:
            return torch.sum(rho * weight) / torch.clamp(torch.sum(weight), min=1.0)
    return torch.mean(rho)

def finetune(param, model, train_data, criterion=None, criterion_fl=None,
             input_attrs=["data"], output_attrs=["label"],
             plot=True, plot_epoch=False, mtl=False, transfer=None, pr_de=False,
             save_data=False, str_ort=False, alp=0.5, facies=False,
             CIGLoss_type='L2', ciglabel_dir=None, use_ep=False,
             frame_part=False, file_name=None):

    # ---------------- 初始化参数 ----------------
    epochs = param['epochs']
    batch_size = param['batch_size']
    lr = param['lr']
    optimizer_type = param['optimizer_type']
    weight_decay = param['weight_decay']
    disp_inter = param['disp_inter']
    save_inter = param['save_inter']
    checkpoint_path = param['checkpoint_path']
    loss_type = param['loss']
    print(f"loss type={loss_type}")
    n1, n2, n3 = param['data_shape']
    ol_fr1 = param['ol_fr1']
    trans_epoch = param['trans_epoch']
    ol_lora = param['ol_lora']
    a1, a2, a3 = param['a3']
    pred_local = param['pred_local']
    use_mx_valid = param['mx_valid']
    facies_3D = param["facies_3D"]
    seg_first = param["seg_first"]

    # 这些是历史遗留键，改 .get() 后即可从 config 安全删除（不删也无害）
    lr_patience = param.get('lr_patience', 8)
    lr_factor   = param.get('lr_factor', 0.5)
    gamma       = param.get('gamma', 0.9)
    step_size   = param.get('step_size', 50)
    momentum    = param.get('momentum', 0.8)

    # ---------------- 新增配置 ----------------
    boundary_weight = param.get('boundary_weight', 0.0)
    boundary_margin = param.get('boundary_margin', 0.5)
    max_depth_val   = param.get('max_depth', 10.0)
    phase_weight = param.get('phase_weight', 0.0)
    phase_amp_percentile = param.get('phase_amp_percentile', 70.0)
    phase_penalty = param.get('phase_penalty', 'l1')            # 'l1' | 'charbonnier' | 'welsch'
    phase_penalty_scale = param.get('phase_penalty_scale', 0.3)
    phase_warmup_epochs = param.get('phase_warmup_epochs', 0)   # 前 N epoch 线性 ramp-up，0=不启用
    segment_teacher_weight = param.get('segment_teacher_weight', 0.0)
    segment_teacher_warmup_epochs = param.get('segment_teacher_warmup_epochs', 0)
    seg_order_weight = param.get('seg_order_weight', 0.0)
    seg_order_warmup_epochs = param.get('seg_order_warmup_epochs', 0)
    seg_order_min_points = param.get('seg_order_min_points', 5)
    seg_order_min_depth_gap = param.get('seg_order_min_depth_gap', 4.0)
    seg_order_margin = param.get('seg_order_margin', 0.02)
    seg_order_max_segments = param.get('seg_order_max_segments', 128)
    frame_anchor_weight = param.get('frame_anchor_weight', 0.0)
    consistency_weight = param.get('consistency_weight', 0.0)
    consistency_path = param.get('consistency_path', None)
    consistency_target = None
    consistency_np = None
    # 跨切片成对一致性：batch 由相邻切片成对组成，约束成对预测的差值
    # （与 consistency_weight 的"锚定参考体"机制互斥使用；共用 consistency 打印字段）
    pair_consistency_weight = param.get('pair_consistency_weight', 0.0)
    pair_beta = param.get('pair_beta', 0.02)
    pair_gap = int(param.get('pair_gap', 1))
    pair_depth_gamma = param.get('pair_depth_gamma', 0.0)
    # dip 补偿版 pair：δ 场来自 PWD（make_pair_dip_512.py），比较 pa(z) 与 pb(z+δ)
    # ——同位置版是 δ≡0 特例。仅适用于"样本索引=主方向切片号"的数据集（如纯 xline）。
    pair_dip_path = param.get('pair_dip_path', '')
    pair_dip_scale = float(param.get('pair_dip_scale', 0.0)) or float(pair_gap)
    pair_dip_np = None
    if pair_consistency_weight > 0 and pair_dip_path:
        pair_dip_np = torch.from_numpy(np.load(pair_dip_path).astype(np.float32))  # (y, z, x)
        print(f">>> pair dip 补偿启用: {pair_dip_path} shape={tuple(pair_dip_np.shape)} "
              f"scale={pair_dip_scale}")
    # 跨断层配对锚点：约束 RGT(zl,xl)==RGT(zr,xr)（zr=zl+互相关断距，可为小数）
    fault_pair_weight = param.get('fault_pair_weight', 0.0)
    fault_pair_beta = param.get('fault_pair_beta', 0.03)
    if fault_pair_weight > 0:
        print(f">>> 启用跨断层配对 loss: weight={fault_pair_weight}, beta={fault_pair_beta}"
              f" (锚点来自样本 fault_pairs 键, conf 加权)")
    if consistency_weight > 0:
        if not consistency_path:
            raise ValueError('consistency_weight > 0 requires consistency_path')
        _n1, _n2, _n3 = param['data_shape']
        consistency_np = np.fromfile(consistency_path, dtype=np.single).reshape(_n3, _n2, _n1).transpose(2, 1, 0)
    if boundary_weight > 0:
        print(f">>> 启用边界锚定 loss: weight={boundary_weight}, "
              f"margin={boundary_margin}, max_depth={max_depth_val}")
    if phase_weight > 0:
        print(f">>> 启用 seismic phase alignment loss: weight={phase_weight}, "
              f"amp_percentile={phase_amp_percentile}, penalty={phase_penalty}, "
              f"scale={phase_penalty_scale}, warmup={phase_warmup_epochs}ep")
    if segment_teacher_weight > 0:
        print(f">>> 启用 segment relative teacher loss: weight={segment_teacher_weight}, "
              f"warmup={segment_teacher_warmup_epochs}ep")
    if seg_order_weight > 0:
        print(f">>> 启用 segment relative order loss: weight={seg_order_weight}, "
              f"warmup={seg_order_warmup_epochs}ep, min_points={seg_order_min_points}, "
              f"min_depth_gap={seg_order_min_depth_gap}, margin={seg_order_margin}, "
              f"max_segments={seg_order_max_segments}")
    if frame_anchor_weight > 0:
        print(f">>> 启用 interpreted horizon absolute frame anchor loss: weight={frame_anchor_weight}")
    if consistency_weight > 0:
        print(f">>> 启用 pretrained consistency loss: weight={consistency_weight}, path={consistency_path}")

    # ---------------- RGT 层位质控（只评价，不参与 loss）----------------
    qc = None
    qc_history = []
    best_qc_deep = 1e30
    qc_gh_path = param.get('qc_gh_path', None)
    if qc_gh_path:
        from rgt_qc import RGTQualityControl
        qc = RGTQualityControl(qc_gh_path,
                               data_shape=param['data_shape'],
                               slice_step=param.get('qc_slice_step', 32),
                               levels_n=param.get('qc_levels', 40),
                               pred_local=param.get('pred_local', 'xline'))
        print(f">>> 启用 RGT 层位质控: ref={qc_gh_path}, 切片数={len(qc.slices)}, "
              f"随 plot_epoch 周期评估，深部指标新低时存 checkpoint-best-qc.pth")

    # 分组 LR / 调度相关
    lr_conv        = param.get('lr_conv', None)        # 预训练 conv 组绝对 LR；None 则用 lr
    lr_lora        = param.get('lr_lora', None)        # LoRA 组绝对 LR；None 则用 lr*mult
    lr_lora_mult   = param.get('lr_lora_mult', 2.0)
    warmup_epochs  = param.get('warmup_epochs', 0)
    scheduler_type = param.get('scheduler_type', 'cosine')
    eta_ratio      = param.get('eta_ratio', 0.01)

    criterion_hr_global = hr_loss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if consistency_weight > 0:
        consistency_target = torch.from_numpy(consistency_np * 10.0).float().to(device)

    # struct loss 按 batch_idx 切片 u1/u2/u3，依赖加载顺序，此时禁止 shuffle
    shuffle_train = param.get('shuffle', False)
    if shuffle_train and ('str' in loss_type or 'struct' in loss_type):
        print("[WARN] loss_type 含 str/struct，按 batch_idx 对齐 u1/u2/u3，强制 shuffle=False")
        shuffle_train = False
    if pair_consistency_weight > 0:
        class _PairBatchSampler(torch.utils.data.Sampler):
            """batch = [i1, i1+gap, i2, i2+gap, ...]；每 epoch 采 n//2 对，
            样本吞吐与普通 shuffle loader 相同"""
            def __init__(self, n, batch_size, gap):
                self.n = n
                self.gap = gap
                self.ppb = max(batch_size // 2, 1)
            def __iter__(self):
                starts = np.random.permutation(self.n - self.gap)[: self.n // 2]
                batch = []
                for s in starts:
                    batch += [int(s), int(s) + self.gap]
                    if len(batch) == self.ppb * 2:
                        yield batch
                        batch = []
                if batch:
                    yield batch
            def __len__(self):
                return int(np.ceil((self.n // 2) / self.ppb))
        train_loader = DataLoader(dataset=train_data,
                                  batch_sampler=_PairBatchSampler(len(train_data), batch_size, pair_gap),
                                  num_workers=20)
        print(f">>> 启用跨切片成对一致性: weight={pair_consistency_weight}, "
              f"beta={pair_beta}, gap={pair_gap} (batch 内偶数位=奇数位的邻切片)")
    else:
        train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=shuffle_train,
                                  drop_last=False, num_workers=20)
    print(len(train_loader))

    # ============================================================
    # 优化器 + 调度器：分组 LR（LoRA / 预训练 conv 各自基准）+ warmup + cosine
    # ============================================================
    def _build_param_groups(model, base_lr, lora_mult, wd, lr_conv=None, lr_lora=None):
        """可训练参数分两组：
           - LoRA 旁路：高 LR（绝对值优先 lr_lora，否则 base*mult），不加 weight_decay
           - 解冻的预训练 conv（SR/DWConv）：基准 LR（绝对值优先 lr_conv，否则 base_lr）
        """
        lora_params, conv_params = [], []
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            (lora_params if 'lora' in n else conv_params).append(p)
        _conv_lr = lr_conv if lr_conv is not None else base_lr
        _lora_lr = lr_lora if lr_lora is not None else base_lr * lora_mult
        groups = []
        if conv_params:
            groups.append({'params': conv_params, 'lr': _conv_lr, 'weight_decay': wd})
        if lora_params:
            groups.append({'params': lora_params, 'lr': _lora_lr, 'weight_decay': 0.0})
        return groups

    if optimizer_type == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_type == "Adam":
        print('use adam')
        _p = filter(lambda p: p.requires_grad, model.parameters()) if transfer else model.parameters()
        optimizer = optim.Adam(_p, lr=lr, weight_decay=weight_decay)
    elif optimizer_type == "Adamw":
        print('use adamw')
        if transfer:
            groups = _build_param_groups(model, lr, lr_lora_mult, weight_decay,
                                         lr_conv=lr_conv, lr_lora=lr_lora)
            optimizer = optim.AdamW(groups)  # 各组 lr/wd 已在 groups 内指定
            for i, g in enumerate(optimizer.param_groups):
                print(f"    group{i}: lr={g['lr']:.2e}, wd={g['weight_decay']}, "
                      f"#params={sum(p.numel() for p in g['params']):,}")
        else:
            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == "Adam_sam":
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                               lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

    # ---- 调度器：LambdaLR 返回统一乘子，按比例缩放所有组，保持 LoRA:conv 比例恒定 ----
    if scheduler_type == 'cosine':
        def lr_lambda(epoch):
            if warmup_epochs > 0 and epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs                  # 线性 warmup：0→1
            progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
            progress = min(max(progress, 0.0), 1.0)
            cos = 0.5 * (1.0 + math.cos(math.pi * progress))        # 1→0
            return eta_ratio + (1.0 - eta_ratio) * cos              # 1→eta_ratio
        scheduler = LambdaLR(optimizer, lr_lambda)
        print(f">>> Cosine: warmup={warmup_epochs}ep, eta_ratio={eta_ratio}, T={epochs}")
    elif scheduler_type == 'multistep':
        ms = [int(epochs * 0.4), int(epochs * 0.7), int(epochs * 0.9)]
        scheduler = MultiStepLR(optimizer, milestones=ms, gamma=0.5)
        print(f">>> MultiStep: milestones={ms}, gamma=0.5")
    elif scheduler_type == 'none':
        scheduler = LambdaLR(optimizer, lambda e: 1.0)              # 恒定，不衰减
        print(">>> LR 不衰减（保持各组初始比例）")
    else:
        scheduler = LambdaLR(optimizer, lambda e: eta_ratio + (1 - eta_ratio) *
                             0.5 * (1 + math.cos(math.pi * min(e / max(1, epochs), 1.0))))
        print(f">>> 未知 scheduler_type='{scheduler_type}'，退回 cosine")

    # ---------------- 损失函数 ----------------
    if criterion is None:
        criterion = nn.MSELoss().to(device)
    if facies_3D:
        seg_depth_gamma = param.get('seg_depth_gamma', 0.0)
        seg_cross_slice = param.get('seg_cross_slice', False)
        criterion_facies = SegmentLoss(depth_weight_gamma=seg_depth_gamma,
                                       cross_slice=seg_cross_slice).to(device)
        if seg_cross_slice:
            print(">>> SegmentLoss 跨切片分组: batch 内同全局 ID 共用中心 "
                  "(要求 shuffle=False, batch 为相邻切片)")
        criterion_segment_order = SegmentOrderLoss(
            min_points=seg_order_min_points,
            min_depth_gap=seg_order_min_depth_gap,
            margin=seg_order_margin,
            max_segments=seg_order_max_segments,
        ).to(device)
        if seg_depth_gamma > 0:
            print(f">>> SegmentLoss 深度加权: gamma={seg_depth_gamma} "
                  f"(最深轴权重为最浅轴的 {1.0 + seg_depth_gamma:.1f} 倍)")
    criterion_ce = nn.BCEWithLogitsLoss().to(device)
    criterion_bce = ceb_loss().to(device)
    print('fault_loss use cbe')
    if facies:
        if CIGLoss_type == "L1":
            criterion2_1 = CIGLoss(loss_type="L1", ciglabel_dir=ciglabel_dir).to(device)
        elif CIGLoss_type == "L2":
            criterion2_1 = CIGLoss(loss_type="L2", ciglabel_dir=ciglabel_dir).to(device)
        else:
            print("CIGLoss is None")
    criterion2_2 = NormalLoss().to(device)

    # ---------------- 主循环准备 ----------------
    epoch_loss_train, epoch_loss_valid, epoch_lr = [], [], []
    epoch_loss_fl_train, epoch_loss_fl_valid, epoch_loss_hr_train, epoch_loss_hr_valid, epoch_loss_mx_valid = [], [], [], [], []
    epoch_hz_px = []
    criterion_dice = DiceLoss()
    best_mse = 1e50

    if 'str' in loss_type:
        print(f'n1,n2,n3 = {n1,n2,n3}')
        u1 = np.fromfile(f'../data/{file_name}/u1.dat', dtype=np.single).reshape(n3, n2, n1).transpose(2, 1, 0)
        u2 = np.fromfile(f'../data/{file_name}/u2.dat', dtype=np.single).reshape(n3, n2, n1).transpose(2, 1, 0)
        u3 = np.fromfile(f'../data/{file_name}/u3.dat', dtype=np.single).reshape(n3, n2, n1).transpose(2, 1, 0)
        if use_ep:
            ep = np.fromfile(f'../data/{file_name}/ep.dat', dtype=np.single).reshape(n3, n2, n1).transpose(2, 1, 0)
            ep = torch.from_numpy(ep[:]).to(device)
        if ol_fr1:
            u1, u2, u3 = param['u_overlap']
        u1 = torch.from_numpy(u1[:]).to(device)
        u2 = torch.from_numpy(u2[:]).to(device)
        u3 = torch.from_numpy(u3[:]).to(device)

    w_chamfer = 0.1
    for epoch in range(epochs):
        since = time.time()

        # ---------------- 训练阶段 ----------------
        model.train()
        loss_train_per_epoch = 0
        loss_hr_per_epoch = 0
        hz_px_per_epoch = 0.0
        loss_mx_valid_per_epoch = 0
        loss_facies_per_epoch = 0
        loss_normal_per_epoch = 0
        loss_train_ssim_per_epoch = 0
        loss_train_hr_global_per_epoch = 0
        loss_train_mx_valid_per_epoch = 0
        loss_fl_train_per_epoch = 0
        loss_hr_train_per_epoch = 0
        loss_boundary_per_epoch = 0
        loss_phase_per_epoch = 0
        loss_frame_anchor_per_epoch = 0
        loss_consistency_per_epoch = 0
        loss_segment_order_per_epoch = 0

        for batch_idx, batch_samples in enumerate(train_loader):

            if 'struct' in loss_type or 'str' in loss_type:
                if batch_idx < len(train_loader) - 1:
                    if not str_ort:
                        if use_ep:
                            criterion_str = STRUCTURELossv2(u1[:, :, batch_size*batch_idx:batch_size*(batch_idx+1)],
                                                            u2[:, :, batch_size*batch_idx:batch_size*(batch_idx+1)],
                                                            u3[:, :, batch_size*batch_idx:batch_size*(batch_idx+1)],
                                                            use_ep=use_ep, ep=ep[:, :, batch_size*batch_idx:batch_size*(batch_idx+1)])
                        else:
                            criterion_str = STRUCTURELossv2(u1[:, :, batch_size*batch_idx:batch_size*(batch_idx+1)],
                                                            u2[:, :, batch_size*batch_idx:batch_size*(batch_idx+1)],
                                                            u3[:, :, batch_size*batch_idx:batch_size*(batch_idx+1)], use_ep=use_ep)
                    else:
                        criterion_str_u1 = STRUCTURELossu_rgt(u1[:, :, batch_size*batch_idx:batch_size*(batch_idx+1)])
                        criterion_str_u2 = STRUCTURELossu_rgt(u2[:, :, batch_size*batch_idx:batch_size*(batch_idx+1)])
                        criterion_str_u3 = STRUCTURELossu_rgt(u3[:, :, batch_size*batch_idx:batch_size*(batch_idx+1)])
                elif batch_idx == len(train_loader) - 1:
                    if not str_ort:
                        if use_ep:
                            criterion_str = STRUCTURELossv2(u1[:, :, batch_size*batch_idx:],
                                                            u2[:, :, batch_size*batch_idx:],
                                                            u3[:, :, batch_size*batch_idx:],
                                                            use_ep=use_ep, ep=ep[:, :, batch_size*batch_idx:])
                        else:
                            criterion_str = STRUCTURELossv2(u1[:, :, batch_size*batch_idx:],
                                                            u2[:, :, batch_size*batch_idx:],
                                                            u3[:, :, batch_size*batch_idx:], use_ep=use_ep)
                    else:
                        criterion_str_u1 = STRUCTURELossu_rgt(u1[:, :, batch_size*batch_idx:])
                        criterion_str_u2 = STRUCTURELossu_rgt(u2[:, :, batch_size*batch_idx:])
                        criterion_str_u3 = STRUCTURELossu_rgt(u3[:, :, batch_size*batch_idx:])

            data = batch_samples["seis"]
            data = data.to(device)
            data = Variable(data)
            if not frame_part:
                frame = batch_samples["frame"]
            else:
                frame = batch_samples["frame_part"]

            if facies and not facies_3D:
                normal = batch_samples['normal']
                cigfacies = batch_samples['cigfacies']
                linearity = batch_samples['linearity']
                indexs = batch_samples['index']
                normal, cigfacies, linearity, indexs = (Variable(normal.to(device)), Variable(cigfacies.to(device)),
                                                        Variable(linearity.to(device)), Variable(indexs.to(device)))
            if str_ort:
                normal = batch_samples['normal']
                linearity = batch_samples['linearity']
                normal, linearity = Variable(normal.to(device)), Variable(linearity.to(device))
            frame = Variable(frame.to(device))
            mx = batch_samples["mx_single"]
            mx = Variable(mx.to(device))
            mx = mx.permute(2, 0, 1, 3, 4)

            optimizer.zero_grad()
            if pr_de:
                data = torch.cat((data, data, data), dim=1)
            else:
                data = torch.cat((frame*10, data, data), dim=1)

            if mtl == False:
                if pr_de:
                    target_i = model(data, target*mask, fl_p)
                else:
                    target_i = model(data)
                target_j = target_i

                loss = 0
                l_hr_global = criterion_hr_global(mx, target_j)
                loss_hr_per_epoch += l_hr_global.item()
                hz_px_per_epoch += hz_depth_px_monitor(mx, target_j)

                if trans_epoch:
                    if epoch < trans_epoch:
                        if seg_first:
                            if not facies_3D:
                                if CIGLoss_type == "L1":
                                    loss2_1 = criterion2_1(target_i, indexs)
                                elif CIGLoss_type == "L2":
                                    loss2_1 = 0.01 * criterion2_1(target_i, indexs)
                            elif facies_3D:
                                segments = batch_samples["segments"]
                                segments = Variable(segments.to(device))
                                seg_w = batch_samples.get("segments_w", None)
                                seg_w = Variable(seg_w.to(device)) if seg_w is not None else None
                                loss2_1 = criterion_facies(target_j, segments, weight=seg_w)
                            loss += a3 * loss2_1
                            loss_facies_per_epoch += a3 * loss2_1.item()
                        else:
                            loss += a1 * l_hr_global
                    else:
                        if seg_first:
                            if a1 != 0:
                                loss += a1 * l_hr_global
                        else:
                            if not facies_3D:
                                if CIGLoss_type == "L1":
                                    loss2_1 = criterion2_1(target_i, indexs)
                                elif CIGLoss_type == "L2":
                                    loss2_1 = 0.01 * criterion2_1(target_i, indexs)
                            elif facies_3D:
                                segments = batch_samples["segments"]
                                segments = Variable(segments.to(device))
                                seg_w = batch_samples.get("segments_w", None)
                                seg_w = Variable(seg_w.to(device)) if seg_w is not None else None
                                loss2_1 = criterion_facies(target_j, segments, weight=seg_w)
                            loss += a3 * loss2_1
                            loss_facies_per_epoch += a3 * loss2_1.item()
                elif not trans_epoch:
                    if a1 != 0:
                        loss += a1 * l_hr_global
                    if a3 != 0:
                        if not facies_3D:
                            if CIGLoss_type == "L1":
                                loss2_1 = criterion2_1(target_i, indexs)
                            elif CIGLoss_type == "L2":
                                loss2_1 = 0.01 * criterion2_1(target_i, indexs)
                        elif facies_3D:
                            segments = batch_samples["segments"]
                            segments = Variable(segments.to(device))
                            seg_w = batch_samples.get("segments_w", None)
                            seg_w = Variable(seg_w.to(device)) if seg_w is not None else None
                            loss2_1 = criterion_facies(target_j, segments, weight=seg_w)
                        loss += a3 * loss2_1
                        loss_facies_per_epoch += a3 * loss2_1.item()

                # === 软边界锚定 loss ===
                # 仅在启用了 boundary_weight 且过了 trans_epoch 时生效
                if boundary_weight > 0 and (not trans_epoch or epoch >= trans_epoch):
                    # target_j 形状: [B, 1, H, W]，dim=2 是深度方向
                    # 单侧锚定：top<=margin, bot>=max_depth-margin。
                    # 注意：不要对 bot 加过冲上界！模型初始在最深层位以下的外推区
                    # 会过冲到 ~2.6*max_depth，双侧约束会在早期贡献 ~97% 的 loss
                    # 把 hr 主项压垮（实测 epoch0-5 训练直接崩）。
                    top_row = target_j[:, :, 0, :]
                    bot_row = target_j[:, :, -1, :]
                    l_top = F.relu(top_row - boundary_margin).pow(2).mean()
                    l_bot = F.relu((max_depth_val - boundary_margin) - bot_row).pow(2).mean()
                    l_boundary = l_top + l_bot
                    loss = loss + boundary_weight * l_boundary
                    loss_boundary_per_epoch += l_boundary.item()

                if frame_anchor_weight > 0:
                    anchor_mask = (mx.sum(dim=0) > 0)
                    if anchor_mask.any():
                        l_frame_anchor = F.smooth_l1_loss(target_j[anchor_mask], (frame * 10.0)[anchor_mask], beta=0.05)
                        loss = loss + frame_anchor_weight * l_frame_anchor
                        loss_frame_anchor_per_epoch += l_frame_anchor.item()

                if consistency_weight > 0:
                    idxs = batch_samples['index'].long().to(device)
                    if pred_local == 'xline':
                        consistency_batch = consistency_target[:, idxs, :].permute(1, 0, 2).unsqueeze(1)
                    else:
                        consistency_batch = consistency_target[:, :, idxs].permute(2, 0, 1).unsqueeze(1)
                    l_consistency = F.smooth_l1_loss(target_j, consistency_batch, beta=0.05)
                    loss = loss + consistency_weight * l_consistency
                    loss_consistency_per_epoch += l_consistency.item()

                if pair_consistency_weight > 0:
                    # batch 由 _PairBatchSampler 保证 [i, i+gap] 成对相邻
                    pa = target_j[0::2]
                    pb = target_j[1::2]
                    mnp = min(pa.shape[0], pb.shape[0])
                    if mnp > 0:
                        pb_eff = pb[:mnp]
                        vmask = None
                        if pair_dip_np is not None:
                            # 沿构造方向一致: pb 按 δ 竖直亚像素 warp 后与 pa 同位置比
                            ia = batch_samples['index'].long()[0::2][:mnp]
                            ia = ia.clamp(0, pair_dip_np.shape[0] - 1)
                            d = (pair_dip_np[ia] * pair_dip_scale).to(target_j.device)
                            pbs = pb_eff.squeeze(1)                       # (B2, Z, X)
                            Zd = pbs.shape[1]
                            zt = torch.arange(Zd, device=pbs.device,
                                              dtype=pbs.dtype).view(1, Zd, 1) + d
                            z0 = zt.floor().clamp(0, Zd - 1)
                            z1 = (z0 + 1).clamp(0, Zd - 1)
                            f = (zt - z0).clamp(0, 1)
                            pb_eff = (pbs.gather(1, z0.long()) * (1 - f)
                                      + pbs.gather(1, z1.long()) * f).unsqueeze(1)
                            vmask = ((zt >= 0) & (zt <= Zd - 1)).unsqueeze(1).to(pbs.dtype)
                        pl = F.smooth_l1_loss(pa[:mnp], pb_eff, beta=pair_beta, reduction='none')
                        Hd = pl.shape[2]
                        zc = torch.arange(Hd, device=pl.device, dtype=pl.dtype)
                        # 深度加权（gamma=0 退化为均匀）；与 dip 有效域掩膜相乘归一
                        w = (1.0 + pair_depth_gamma * zc / max(Hd - 1, 1)).view(1, 1, Hd, 1)
                        w = w.expand_as(pl)
                        if vmask is not None:
                            w = w * vmask
                        l_pair = (pl * w).sum() / w.sum().clamp(min=1.0)
                        loss = loss + pair_consistency_weight * l_pair
                        loss_consistency_per_epoch += l_pair.item()

                if fault_pair_weight > 0 and 'fault_pairs' in batch_samples:
                    fp = batch_samples['fault_pairs'].to(device)      # [B, M, 5]
                    conf = fp[..., 4]
                    valid = conf > 0
                    if valid.any():
                        Hd = target_j.shape[2]
                        bidx = torch.arange(fp.shape[0], device=device).unsqueeze(1).expand_as(conf)
                        zl = fp[..., 0].long().clamp(0, Hd - 1)
                        xl = fp[..., 1].long().clamp(0, target_j.shape[3] - 1)
                        xr = fp[..., 3].long().clamp(0, target_j.shape[3] - 1)
                        zr = fp[..., 2].clamp(0, Hd - 1 - 1e-3)
                        z0 = zr.floor().long()
                        fz = (zr - z0.float())
                        p = target_j[:, 0]
                        vl = p[bidx, zl, xl]
                        vr = p[bidx, z0, xr] * (1 - fz) + p[bidx, (z0 + 1).clamp(max=Hd - 1), xr] * fz
                        diff = vl - vr
                        ad = diff.abs()
                        pl = torch.where(ad < fault_pair_beta,
                                         0.5 * diff ** 2 / fault_pair_beta,
                                         ad - 0.5 * fault_pair_beta)
                        wsum = (conf * valid).sum()
                        l_fp = (pl * conf * valid).sum() / torch.clamp(wsum, min=1.0)
                        loss = loss + fault_pair_weight * l_fp
                        loss_frame_anchor_per_epoch += l_fp.item()

                if phase_weight > 0:
                    # 训练初期预测尚乱，切向方向不可靠 → 线性 ramp-up
                    if phase_warmup_epochs > 0:
                        w_phase = phase_weight * min(1.0, (epoch + 1) / phase_warmup_epochs)
                    else:
                        w_phase = phase_weight
                    l_phase = seismic_phase_alignment_loss(
                        target_j, batch_samples["seis"].to(device),
                        amp_percentile=phase_amp_percentile,
                        penalty=phase_penalty, penalty_scale=phase_penalty_scale)
                    loss = loss + w_phase * l_phase
                    loss_phase_per_epoch += l_phase.item()

                if seg_order_weight > 0 and facies_3D:
                    segments_order = Variable(batch_samples["segments"].to(device))
                    if seg_order_warmup_epochs > 0:
                        w_seg_order = seg_order_weight * min(1.0, (epoch + 1) / seg_order_warmup_epochs)
                    else:
                        w_seg_order = seg_order_weight
                    l_seg_order = criterion_segment_order(target_j, segments_order)
                    loss = loss + w_seg_order * l_seg_order
                    loss_segment_order_per_epoch += w_seg_order * l_seg_order.item()

                if segment_teacher_weight > 0 and 'segments_teacher' in batch_samples:
                    teacher = batch_samples['segments_teacher'].to(device)
                    mask_t = teacher > 0
                    if mask_t.any():
                        if segment_teacher_warmup_epochs > 0:
                            w_teacher = segment_teacher_weight * min(1.0, (epoch + 1) / segment_teacher_warmup_epochs)
                        else:
                            w_teacher = segment_teacher_weight
                        l_seg_teacher = F.smooth_l1_loss(target_j[mask_t], teacher[mask_t], beta=0.02)
                        loss = loss + w_teacher * l_seg_teacher
                        loss_facies_per_epoch += w_teacher * l_seg_teacher.item()

                if use_mx_valid:
                    mx_valid = batch_samples["mask_valid"]
                    mx_valid = Variable(mx_valid.to(device))
                    mx_valid = mx_valid.permute(2, 0, 1, 3, 4)
                    l_mx_valid = criterion_hr_global(mx_valid, target_j)
                    loss_mx_valid_per_epoch += l_mx_valid.item()

                # NaN 检查必须在 step 之前，否则坏梯度已经写进权重
                if not torch.isfinite(loss):
                    print(f'[WARN] epoch {epoch} batch {batch_idx}: loss={loss.item()}, 跳过本次更新')
                    optimizer.zero_grad()
                else:
                    loss.backward()
                    optimizer.step()
                    loss_train_per_epoch += loss.item()

        # ---------------- 周期性可视化/保存 ----------------
        if plot_epoch:
            if epoch % plot_epoch == 0 or epoch == epochs - 1:
                pred_samples = pred_dict_2d23d(model, train_data)

                # --- RGT 层位质控（复用本次全量推理，零额外成本）---
                if qc is not None:
                    qc_m = qc.evaluate(pred_samples)
                    qc_history.append({'epoch': epoch, **qc_m})
                    print(">>> QC[ep{}] 层位偏差(px): all={:.2f} deep={:.2f} bands={}".format(
                        epoch, qc_m['all'], qc_m['deep'],
                        '/'.join('{:.1f}'.format(v) for v in qc_m['bands'])))
                    if qc_m['deep'] < best_qc_deep:
                        best_qc_deep = qc_m['deep']
                        state = {'epoch': epoch, 'state_dict': model.state_dict(),
                                 'optimizer': optimizer.state_dict()}
                        torch.save(state, os.path.join(checkpoint_path, 'checkpoint-best-qc.pth'))
                        print(f">>> QC 深部指标新低 ({best_qc_deep:.2f}px)，已存 checkpoint-best-qc.pth")

                _np = len(pred_samples)
                sn = [min(50, _np - 1), min(100, _np - 1), _np - 1]
                select_samples_2 = []
                for i in range(len(sn)):
                    select_samples_2.append(pred_samples[sn[i]])
                savep_path = os.path.join(checkpoint_path, 'png')
                if not os.path.exists(savep_path):
                    os.makedirs(savep_path)
                save_file = f'{savep_path}_{epoch}.png'

                if save_data == True:
                    saved_path = os.path.join(checkpoint_path, 'dat')
                    if not os.path.exists(savep_path):
                        os.makedirs(savep_path)
                    saved_file = f'{savep_path}_{epoch}.dat'
                    rgt_pred_in = np.zeros((n1, n2, n3), dtype=np.single)
                    # 混合方向数据集样本数可超过体切片数：前 n2/n3 个样本按约定
                    # 是主方向切片的顺序排列，多出的样本只参与训练不参与回填
                    if pred_local == 'inline':
                        for i in range(min(len(pred_samples), n3)):
                            rgt_pred_in[:, :, i] = pred_samples[i]['pred'][0]
                    elif pred_local == 'xline':
                        for i in range(min(len(pred_samples), n2)):
                            rgt_pred_in[:, i, :] = pred_samples[i]['pred'][0]
                    rgt_pred_in = rgt_pred_in.transpose()
                    rgt_pred_in.tofile(saved_file)

                if facies:
                    if facies_3D:
                        draw_samples(select_samples_2[:], attr_list=['frame', 'hrzs2', 'pred', 'cpred', 'segments'],
                                     save=True, save_file=save_file, fl_max=0.15, seis_min=-2, seis_max=2)
                    else:
                        draw_samples(select_samples_2[:], attr_list=['frame', 'hrzs2', 'pred', 'cpred', 'cigfacies'],
                                     save=True, save_file=save_file, fl_max=0.15, seis_min=-2, seis_max=2)
                else:
                    draw_samples(select_samples_2[:], attr_list=['frame', 'hrzs2', 'pred', 'cpred'],
                                 save=True, save_file=save_file, fl_max=0.15)

        # ---------------- 每 epoch 汇总 ----------------
        loss_train_per_epoch = loss_train_per_epoch / len(train_loader)
        loss_train_ssim_per_epoch = loss_train_ssim_per_epoch / len(train_loader)
        loss_train_hr_global_per_epoch = loss_hr_per_epoch / len(train_loader)
        loss_boundary_per_epoch = loss_boundary_per_epoch / len(train_loader)
        loss_phase_per_epoch = loss_phase_per_epoch / len(train_loader)
        loss_segment_order_per_epoch = loss_segment_order_per_epoch / len(train_loader)
        if use_mx_valid:
            loss_train_mx_valid_per_epoch = loss_mx_valid_per_epoch / len(train_loader)
            epoch_loss_mx_valid.append(loss_train_mx_valid_per_epoch)
        if facies:
            loss_facies_per_epoch = loss_facies_per_epoch / len(train_loader)
            loss_nromal_per_epoch = loss_normal_per_epoch / len(train_loader)
        if 'hr' in loss_type and 'struct' in loss_type:
            loss_normal_per_epoch = loss_normal_per_epoch / len(train_loader)

        epoch_loss_train.append(loss_train_per_epoch)
        epoch_hz_px.append(hz_px_per_epoch / len(train_loader))
        epoch_lr.append(optimizer.param_groups[0]['lr'])  # 记录 group0(conv) 的 LR

        # 保存模型
        if epoch % save_inter == 0 or epoch == epochs - 1:
            state = {'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
            filename = os.path.join(checkpoint_path, 'checkpoint-epoch{}.pth'.format(epoch))
            if ol_lora:
                torch.save(lora.lora_state_dict(model), filename)
            else:
                torch.save(state, filename)

        # 保存最优模型
        if loss_train_per_epoch < best_mse:
            state = {'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
            filename = os.path.join(checkpoint_path, 'checkpoint-best.pth')
            if ol_lora:
                torch.save(lora.lora_state_dict(model), filename)
            else:
                torch.save(state, filename)
            best_mse = loss_train_per_epoch

        scheduler.step()
        time_elapsed = time.time() - since

        # 显示 loss
        if epoch % disp_inter == 0:
            if not mtl:
                if facies:
                    print('Epoch:{}, Training Loss:{:.8f} hr:{:.8f} facies:{:.8f} boundary:{:.8f} phase:{:.8f} seg_order:{:.8f} frame_anchor:{:.8f} consistency:{:.8f} mx_valid:{:.8f} Learning rate: {:.8f} time:{:.0f}m {:.0f}s'.format(
                        epoch, loss_train_per_epoch, loss_train_hr_global_per_epoch, loss_facies_per_epoch,
                        loss_boundary_per_epoch, loss_phase_per_epoch, loss_segment_order_per_epoch, loss_frame_anchor_per_epoch, loss_consistency_per_epoch,
                        loss_train_mx_valid_per_epoch, epoch_lr[epoch], time_elapsed // 60, time_elapsed % 60))
                elif 'hr' in loss_type and 'struct' in loss_type:
                    print('Epoch:{}, Training Loss:{:.8f} hr:{:.8f} normal:{:.8f} Learning rate: {:.8f} time:{:.0f}m {:.0f}s'.format(
                        epoch, loss_train_per_epoch, loss_train_hr_global_per_epoch, loss_normal_per_epoch,
                        epoch_lr[epoch], time_elapsed // 60, time_elapsed % 60))

    # ---------------- 训练曲线 ----------------
    if plot:
        x = [i for i in range(epochs)]
        fig = plt.figure(figsize=(12, 4))
        ax = fig.add_subplot(1, 2, 1)
        ax.plot(x, smooth(epoch_loss_train, 0.6), label='Training loss')
        ax.set_xlabel('Epoch', fontsize=15)
        ax.set_ylabel('Loss', fontsize=15)
        ax.set_title('Training curve', fontsize=15)
        ax.grid(True)
        plt.legend(loc='upper right', fontsize=15)

        ax = fig.add_subplot(1, 2, 2)
        ax.plot(x, epoch_lr, label='Learning Rate')
        ax.set_xlabel('Epoch', fontsize=15)
        ax.set_ylabel('Learning Rate', fontsize=15)
        ax.set_title('Learning rate curve', fontsize=15)
        ax.grid(True)
        plt.legend(loc='upper right', fontsize=15)
        plt.show()

    logs = {"epoch_loss_train": epoch_loss_train, "epoch_lr": epoch_lr,
            "epoch_hz_px": epoch_hz_px}
    if use_mx_valid:
        logs["epoch_loss_mx_valid"] = epoch_loss_mx_valid
    np.save(os.path.join(checkpoint_path, 'logs.npy'), logs)
    print("logs saved in " + os.path.join(checkpoint_path, 'logs.npy'))

    return model

def plot_facie(filename,slice_num,nx,nt):

    facies_data = []
    for num in slice_num:
        data = np.load(f"../data/{filename}/seeds/"+str(num)+".npy",allow_pickle=True).item()

        seeds = data["seeds"].astype(np.int64)
        path = data["seeds_paths"].astype(np.int64)
        sl = np.fromfile(f"../data/{filename}/train_slope/"+str(num)+".dat",dtype=np.float32).reshape((nx,nt)).T

        sx = np.fromfile(f"../data/{filename}/train/"+str(num)+".dat",dtype=np.float32).reshape((nx,nt)).T
        fx = np.fromfile(f"../data/{filename}/train_cigfacies_enhance//"+str(num)+".dat",dtype=np.float32).reshape((nx,nt)).T

        # data = np.load("./dataset/mhe/data/2d/Beagle_seeds/"+num+".npy",allow_pickle=True).item()
        # seeds = data["seeds"].astype(np.int64)
        # path = data["seeds_paths"].astype(np.int64)
        # sl = np.fromfile("./dataset/mhe/data/2d/Beagle_slope/"+num+".dat",dtype=np.float32).reshape((520,300)).T
        # fx = np.fromfile("./dataset/mhe/data/2d/Beagle_cigfacies_enhance/"+num+".dat",dtype=np.float32).reshape((520,300)).T
        # plotting
        sn = len(seeds)
        # sn = 500
        h,w = fx.shape
        width = 10
        data_s = np.zeros_like(fx)
        data_p = np.zeros_like(fx)
        data_p2 = np.zeros_like(fx)
        vi = 1
        sort_rule = np.arange(0,len(seeds),1)
        np.random.shuffle(sort_rule)
        sort_rule = sort_rule[0:sn]
        seeds = seeds[sort_rule]
        path = path[sort_rule]

        for si in range(sn):
            xi,yi = seeds[si]
            pi = path[si]
            data_s[xi,yi] = 1
            x1,x2 = max(0,xi-width),min(h,xi+width+1) # local windows
            y1,y2 = max(0,yi-width),min(w,yi+width+1)
            px,py = np.where(pi==1)
            for i in range(len(px)):
                data_p2[px[i]+x1,py[i]+y1] += 1
            vi = vi+1
        data_p2[data_p2==0] = np.nan
        facies_data.append(data_p2)
    return(facies_data)



# 模型推理
def pred_dict_2d23d(model, samples, input_attrs=["data"], output_attrs=["label"],values=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    pred_samples = []

    with torch.no_grad(): 
        for i, sample_pred in enumerate(samples):     

            data = mea_std_norm(sample_pred["seis"])
            data = torch.from_numpy(data).unsqueeze(0).float()
            frame = (sample_pred["frame"])
            frame = torch.from_numpy(frame).unsqueeze(0).float()        
    
                        
            
            data, frame = data.to(device), frame.to(device)
            data, frame = Variable(data), Variable(frame)
#             data = torch.cat((data, data,data), dim=1)
            data = torch.cat((frame*10, data,data), dim=1)

            target_hr= model(data) 


            target_hr = target_hr.cpu().squeeze(0).numpy()   
            
            sample_pred["pred"] = target_hr/10
            sample_pred["frame"] =  sample_pred['frame']
            # sample_pred["fr_mean"] =  sample_pred['fr_mean']


            pred_samples.append(sample_pred)
            
    return pred_samples

def pred_dict(model, samples, input_attrs=["data"], output_attrs=["label"]):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    pred_samples = []
    
    with torch.no_grad(): 
        for i, sample_pred in enumerate(samples):     
         
#             for i, input_attr in enumerate(input_attrs):
#                 tmp = sample_pred[input_attr]
#                 tmp = torch.from_numpy(tmp).unsqueeze(0).float()
#                 if i  == 0:
#                     data = tmp     
#                 else:       
#                     data = torch.cat((data, tmp), dim=1

            data = sample_pred["fault"] 
            data = torch.from_numpy(data).unsqueeze(0).float()
        
            for i, output_attr in enumerate(output_attrs):
                tmp = sample_pred[output_attr]
                tmp = torch.from_numpy(tmp).unsqueeze(0).float()
                if i  == 0:
                    target = tmp
                else:
                    target = torch.cat((target, tmp), dim=1)       
                        
            sample_pred['mask'] = sample_pred['frame'].astype(np.bool_).astype(np.single)
            mask = torch.from_numpy(sample_pred['mask']).unsqueeze(0).float()
            
            data, target, mask = data.to(device), target.to(device), mask.to(device)
            data, target, mask = Variable(data), Variable(target), Variable(mask)
            
            data = torch.cat((target * mask, data), dim=1)
            target_i = model(data)    
        
            target_j =  target_i * (1 - mask) + target * mask
#             target_j = target_i
            
            target_j = target_j.cpu().squeeze(0).numpy()
            
            sample_pred["pred"] = target_j
                
            pred_samples.append(sample_pred)
    return pred_samples

def get_train_sample_from_rgt(rx, possible_num_hrzs, hrz_grp, bit=256, sample_rate=2, fl=None):
    h, w = rx.shape
    if fl is None:
        fl = np.zeros(rx.shape)

    num_hrzs = random.choice(possible_num_hrzs)
    num_valid_hrzs = bit

    ux = min_max_norm(rx)
    ux = ux * (num_valid_hrzs - 1)

    valid_hrzs_idxs = sorted(np.unique(np.around(ux)).tolist())[12:-12]
    valid_hrzs_idxs = [d for i, d in enumerate(valid_hrzs_idxs) if i % sample_rate == 0]

    if len(valid_hrzs_idxs) < num_hrzs:
        num_hrzs = max(1, len(valid_hrzs_idxs))

    itv_js = max(1, int(len(valid_hrzs_idxs) / num_hrzs))
    hrzs_idxs = []
    for j in range(num_hrzs - 1):
        start = j * itv_js
        end = (j + 1) * itv_js
        if start < len(valid_hrzs_idxs) and end <= len(valid_hrzs_idxs):
            hrzs_idxs += random.sample(valid_hrzs_idxs[start:end], 1)

    start = (num_hrzs - 1) * itv_js
    if start < len(valid_hrzs_idxs):
        hrzs_idxs += random.sample(valid_hrzs_idxs[start:], 1)

    fx = np.zeros(ux.shape, dtype=np.single)
    mx = np.zeros((len(hrzs_idxs), h, w), dtype=np.single)

    for k, hrzs_idx in enumerate(hrzs_idxs):
        x, y = np.where((ux >= hrzs_idx - sample_rate/2) & (ux < (hrzs_idx + sample_rate/2)))
        for i in range(len(x)):
            if fl[x[i], y[i]] > 0:
                continue
            fx[x[i], y[i]] = ux[x[i], y[i]]
            mx[k, x[i], y[i]] = 1

    return fx / (bit - 1), ux / (bit - 1), mx

def pred_test(model, samples, input_attrs=["data"], output_attrs=["label"]):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    pred_samples = []
    
    with torch.no_grad(): 
        for i, sample_pred in enumerate(samples):     

            data = sample_pred["fault"] 
            data = torch.from_numpy(data).unsqueeze(0).float()
        
            for i, output_attr in enumerate(output_attrs):
                tmp = sample_pred[output_attr]
                tmp = torch.from_numpy(tmp).unsqueeze(0).float()
                if i  == 0:
                    target = tmp
                else:
                    target = torch.cat((target, tmp), dim=1)       
                        
            frame = torch.from_numpy(sample_pred['frame']).unsqueeze(0).float()
            
            data, target, frame = data.to(device), target.to(device), frame.to(device)
            data, target, frame = Variable(data), Variable(target), Variable(frame)
            
            data = torch.cat((frame, data), dim=1)
            target_i = model(data)    
        
            target_j = target_i
            
            target_j = target_j.cpu().squeeze(0).numpy()
            
            sample_pred["pred"] = target_j
                
            pred_samples.append(sample_pred)
    return pred_samples