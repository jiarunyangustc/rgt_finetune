import os
import math
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim

from skimage import measure
from scipy.interpolate import interp1d, interp2d, griddata
from scipy.ndimage import gaussian_filter

def separate_hrzs(fx, ux, bit, sample_rate):

    hrz_idxs = np.arange(0,bit-1,sample_rate)

    hrzs_g = []
    for hrz_idx in hrz_idxs[1:-1]:
        x, y = np.where((fx*(bit-1) >= hrz_idx - sample_rate/2) & (fx*(bit-1) < hrz_idx + sample_rate/2))
        if len(x):
            hrzs_g.append([x,y])


    hvs,hms = [],[]

    for i in range(len(hrzs_g)):
        x,y = hrzs_g[i]

        hm = np.zeros(fx.shape)
        for i in range(len(x)):
            hm[x[i]][y[i]] = 1.0

        hv = np.sum(ux*hm)/hm.sum()

        hvs.append(hv)
        hms.append(hm)

    return hvs, hms

def cut_lins_hors(lines):
    hrzs = []
    for line in lines:
        if len(line[0]) == 0:
            continue
        line = np.array(line)
        idxs = np.argsort(line[1])
        i1s = line[0][idxs]
        i2s = line[1][idxs]
        ib = 0
        for i in range(len(i2s)-1):
            if i2s[i+1]>i2s[i]+1:
                hrzs.append([i1s[ib:i+1],i2s[ib:i+1]])
                ib = i+1
        hrzs.append([i1s[ib:i+1],i2s[ib:i+1]])
    return hrzs

def extract_horizon_img2d(ux,uv):
    if isinstance(uv,list):
        uv = np.array(uv)
    n1,n2 = ux.shape
    vx = np.zeros((len(uv),n2))
    for i2 in range(n2):
        c0 = ux[:,i2]
        for i,v in enumerate(uv):
            if v <= np.max(c0) and v >= np.min(c0):
                c0, x0 = np.unique(c0, return_index=True)
                i1 = np.argsort(c0, axis=0)
                x0 = np.take_along_axis(x0, i1, axis=0)
                c0 = np.take_along_axis(c0, i1, axis=0)
                c0 = gaussian_filter(c0, sigma=3)
                x0 = gaussian_filter(x0, sigma=3)
                f = interp1d(c0, x0, fill_value="extrapolate")
                vx[i,i2] = f(v)
    return vx



def get_hrzs_from_volume(ux, hv):
    if isinstance(hv, list):
        hv = np.array(hv)
    n1,n2,n3 = ux.shape
    hrzs = np.zeros((len(hv),n2,n3))
    for i3 in range(n3):
        for i2 in range(n2):
            c0 = ux[:,i2,i3]
            for i,v in enumerate(hv):
                if v <= np.max(c0) and v >= np.min(c0):
                    c0, x0 = np.unique(c0, return_index=True)
                    i1 = np.argsort(c0, axis=0)
                    x0 = np.take_along_axis(x0, i1, axis=0)
                    c0 = np.take_along_axis(c0, i1, axis=0)
                    c0 = gaussian_filter(c0, sigma=4)
                    x0 = gaussian_filter(x0, sigma=4)
                    f = interp1d(c0, x0, fill_value=0, bounds_error=False)
                    hrzs[i,i2,i3] = f(v)
    return hrzs

def find_near_list(a, n=1e1):
    a = np.array(a)
    b = np.around(a*n)
    c = np.unique(b)
    d = list()
    for i in range(len(c)):
        idxs = np.where(b==c[i])
        d.append(a[idxs].mean())
    return d

def compute_in_hrzs(fx, hms):

    hrzs = []
    for hm in hms:
        fm = fx * hm
        i1s,i2s = [],[]
        n1,n2 = fm.shape
        for i2 in range(n2):
            tmp = np.where(fm[:,i2]>0)[0]
            if len(tmp):
                i1s.append(tmp.mean())
                i2s.append(i2)
        hrzs.append([i1s,i2s])

    hrzs = cut_lins_hors(hrzs)

    return hrzs
def Horizon_extraction_error_syndata(rgt,pred,values):
#     value = np.zeros(mx_single.shape[0],dtype = np.single)
    n1,n2 = pred.shape


    hz_syn = extract_horizon_img2d(rgt,values)

    x = np.linspace(0,n1-1,n1)
    y_new_sum = np.zeros((len(values),n2),dtype = np.single)
    for i in range(n2):
        y = pred[:,i]
        f = interp1d(x,y,kind='cubic',fill_value="extrapolate")
        y_new = f(hz_syn[:,i])
        for j in range(len(values)):
            if n1-1>hz_syn[j,i]>0:
                y_new_sum[j,i] = y_new[j]

    values_pred=[]
    for i in range(len(y_new)):
        sum_len = np.where(y_new_sum[i]!=0)
        values_pred.append(np.sum(y_new_sum[i,:])/len(sum_len[0]))
    hz_pred = extract_horizon_img2d(pred,values_pred)
    hr_ex_error = []
    js = []
    for i in range(len(values)):
        x1 = np.where(hz_syn[i] >0)
        x2 = np.where(hz_syn[i] <n1-1)
        y1 = np.where(hz_pred[i] >0)
        y2 = np.where(hz_pred[i] <n1-1)
        lc = x1[0].tolist()+x2[0].tolist()+y1[0].tolist()+y2[0].tolist()
        lc= list(set(x1[0]).intersection(set(x2[0])).intersection(set(y1[0])).intersection(set(y2[0])))

        hr_ex_error.append(np.sum(abs(hz_syn[i,lc]-hz_pred[i,lc]))/(len(lc)))
        js.append(len(lc))
    hr_ex_error_all=0
    for i in range(len(hr_ex_error)):
        print('horizon {} value:{} hee= {}'.format(i+1,values[i],hr_ex_error[i]))
        hr_ex_error_all += hr_ex_error[i]*js[i]
    hr_ex_error_all = hr_ex_error_all/(sum(js))
    print(f'hee_all = {np.sum(hr_ex_error_all)}')
    return hr_ex_error_all,sum(js)

def Horizon_extraction_error_extra(fr_extra,pred):
#     value = np.zeros(mx_single.shape[0],dtype = np.single)


    n1,n2 = pred.shape
    hz_syn = fr_extra

    x = np.linspace(0,n1-1,n1)
    y_new_sum = np.zeros((len(hz_syn),n2),dtype = np.single)
    for i in range(n2):
        y = pred[:,i]
        f = interp1d(x,y,kind='cubic',fill_value="extrapolate")
        y_new = f(hz_syn[:,i])
        for j in range(len(hz_syn)):
            if n1-1>hz_syn[j,i]>0:
                y_new_sum[j,i] = y_new[j]

    values_pred=[]
    for i in range(len(y_new)):
        sum_len = np.where(y_new_sum[i]!=0)
        values_pred.append(np.sum(y_new_sum[i,:])/len(sum_len[0]))
    hz_pred = extract_horizon_img2d(pred,values_pred)
    hr_ex_error = []
    js = []
    for i in range(len(hz_syn)):
        x1 = np.where(hz_syn[i] >0)
        x2 = np.where(hz_syn[i] <n1-1)
        y1 = np.where(hz_pred[i] >0)
        y2 = np.where(hz_pred[i] <n1-1)
        lc = x1[0].tolist()+x2[0].tolist()+y1[0].tolist()+y2[0].tolist()
        lc= list(set(x1[0]).intersection(set(x2[0])).intersection(set(y1[0])).intersection(set(y2[0])))

        hr_ex_error.append(np.sum(abs(hz_syn[i,lc]-hz_pred[i,lc]))/(len(lc)))
        js.append(len(lc))
    hr_ex_error_all=0
    for i in range(len(hr_ex_error)):
        print('extra horizon {}  hee= {}'.format(i+1,hr_ex_error[i]))
        hr_ex_error_all += hr_ex_error[i]*js[i]
    hr_ex_error_all_pr = hr_ex_error_all/(sum(js))
    print(f'extra hee_all = {np.sum(hr_ex_error_all_pr)}')
    return hr_ex_error_all,sum(js)

def Horizon_extraction_error(mx_single,pred):
#     value = np.zeros(mx_single.shape[0],dtype = np.single)
    n1,n2 = pred.shape
    js = 0
    hee_all = 0
    hee_si = np.zeros(mx_single.shape[0])
    js_si = np.zeros(mx_single.shape[0])

    for i in range(mx_single.shape[0]):
#         hz_ex_me = np.zeros(mx_single.shape[1:],dtype = np.single)
        x,y = np.where(mx_single[i]>0)
        value = []
        value.append(np.sum(x)/(len(x)*127))
        hz_ex = extract_horizon_img2d(pred,value)
#         for j in range(n2):
#             hz_ex_me[round(hz_ex[0][j]),j] = 1
#         x2,y2 = np.where(hz_ex_me>0)
#         for k in range(n2):
#             if hz_ex_me[:,k].any() != 0 and hz_ex_me[:,k].any() != 0:
#                 hz_dis = hz_ex_me[:,k]


        for k in range(len(y)):
            hee_si[i] += abs(x[k] - hz_ex[0][y[k]])
            js_si[i] += 1
        hee_single_mean = hee_si[i]/js_si[i]
        print('hee_single_mean {} = {}'.format(i,hee_single_mean))
    hee_all_mean = np.sum(hee_si)/np.sum(js_si)

    print(f'hee_all_mean = {hee_all_mean}')
    return hee_all_mean,np.sum(hee_si),np.sum(js_si)


def Horizon_extraction_error_2d23d(mx_single,pred,value_fr):
#     value = np.zeros(mx_single.shape[0],dtype = np.single)
    n1,n2 = pred.shape
    js = 0
    hee_all = 0
    hee_si = np.zeros(mx_single.shape[0])
    js_si = np.zeros(mx_single.shape[0])

    for i in range(mx_single.shape[0]):
#         hz_ex_me = np.zeros(mx_single.shape[1:],dtype = np.single)
        x,y = np.where(mx_single[i]>0)
        value = []
        value.append(value_fr[i])
        hz_ex = extract_horizon_img2d(pred,value)
#         for j in range(n2):
#             hz_ex_me[round(hz_ex[0][j]),j] = 1
#         x2,y2 = np.where(hz_ex_me>0)
#         for k in range(n2):
#             if hz_ex_me[:,k].any() != 0 and hz_ex_me[:,k].any() != 0:
#                 hz_dis = hz_ex_me[:,k]


        for k in range(len(y)):
            hee_si[i] += abs(x[k] - hz_ex[0][y[k]])
            js_si[i] += 1
        hee_single_mean = hee_si[i]/js_si[i]
        print('hee_single_mean {} = {}'.format(i,hee_single_mean))
    hee_all_mean = np.sum(hee_si)/np.sum(js_si)

    print(f'hee_all_mean = {hee_all_mean}')
    return hee_all_mean,np.sum(hee_si),np.sum(js_si)


def Horizon_extraction_error_2d23d_float(rgt_lc,pred,value_fr):
#     value = np.zeros(mx_single.shape[0],dtype = np.single)
    print(rgt_lc.shape)
    n1,n2 = pred.shape
    js = 0
    hee_all = 0
    hee_si = np.zeros(len(rgt_lc))
    js_si = np.zeros(len(rgt_lc))

    for i in range(len(rgt_lc)):
#         hz_ex_me = np.zeros(mx_single.shape[1:],dtype = np.single)
        x = np.where(rgt_lc[i]>0)
#         print(len(x))
#         print(len(x[0]))
        value = []
        value.append(value_fr[i])
        hz_ex = extract_horizon_img2d(pred,value)

        for k in range(len(x[0])):
#             print(x[0][k])
            hee_si[i] += abs(rgt_lc[i][x[0][k]] - hz_ex[0][x[0][k]])
            js_si[i] += 1
        hee_single_mean = hee_si[i]/js_si[i]
        print('hee_single_mean {} = {}'.format(i,hee_single_mean))
    hee_all_mean = np.sum(hee_si)/np.sum(js_si)

    print(f'hee_all_mean = {hee_all_mean}')
    return hee_all_mean,np.sum(hee_si),np.sum(js_si)
def compute_out_hrzs(rx, hvs):
    n1 = rx.shape[0]
    hds = extract_horizon_img2d(rx,hvs)

    hrzs = []
    for i,hd in enumerate(hds):
        n = len(hd)
        i1s,i2s = [],[]
        for i2 in range(n):
            if hd[i2]>0 and hd[i2]<n1-1:
                i1s.append(hd[i2])
                i2s.append(i2)
        hrzs.append([i1s,i2s])
    hrzs = cut_lins_hors(hrzs)
    return hrzs

def sampling_density(img, sampling_rate):
    x, y = np.where(img > 0)
    sampleing_num = round(len(x) * sampling_rate)
    d = np.random.choice(len(x), sampleing_num, replace=False)
    img_new = np.zeros(img.shape)
    for i in d:
        img_new[x[i],y[i]] = img[x[i],y[i]]
    return img_new

def compute_hrzs_not_on_grid(fx, hms):

    hrzs = []
    for hm in hms:
        fm = fx * hm
        i1s,i2s = [],[]
        n1,n2 = fm.shape
        for i2 in range(n2):
            tmp = np.where(fm[:,i2]>0)[0]
            if len(tmp):
                i1s.append(tmp.mean())
                i2s.append(i2)
        hrzs.append([i1s,i2s])
    return cut_lins_for_each_hrz(hrzs)

def cut_lins_for_each_hrz(lines):
    hrzs = []
    for line in lines:
        hrz = []
        if len(line[0]) <= 1:
            continue
        line = np.array(line)
        idxs = np.argsort(line[1])
        i1s = line[0][idxs]
        i2s = line[1][idxs]
        ib = 0
        for i in range(len(i2s)-1):
            if i2s[i+1]>i2s[i]+4:
                hrz.append([i1s[ib:i+1],i2s[ib:i+1]])
                ib = i+1
        hrz.append([i1s[ib:i+1],i2s[ib:i+1]])
        hrzs.append(hrz)
    return hrzs

def compute_edist(p0, p1):
    return ((p0[0]-p1[0])**2 + (p0[1]-p1[1])**2)**0.5

def pertb_fals(fals_z):
    fals_new = []

    for j,(i1z,shift_t,shift_b) in enumerate(fals_z):
        n = len(i1z)

        i1s,i2s = np.zeros(n),np.zeros(n)
        rate = (random.uniform(shift_t[0],shift_b[0])-shift_t[0])/(shift_b[0]-shift_t[0])

        if random.random() > 0.5:
            f = interp1d([0, n-1],
                         [shift_t[0] + rate * (shift_b[0]-shift_t[0]),
                          (shift_b[n-1] + shift_t[n-1])/2 + ((shift_b[n-1]-shift_t[n-1])/2 -
                                                           rate * (shift_b[n-1]-shift_t[n-1]))],
                          kind='linear')
            for i in range(n):
                i2s[i] = f(i1z[i])
                i1s[i] = i1z[i]
        else:
            for i in range(n):
                i2s[i] = shift_t[i] + rate * (shift_b[i]-shift_t[i])
                i1s[i] = i1z[i]

        fals_new.append((i1s,i2s))

    return fals_new

def pertb_hrzs(hrzs_z):
    hrzs_new = []

    if random.random() > 0.5:
        sect = True
    else:
        sect = False

    if random.random() < 0.5:
        big = True
    else:
        big = False

    for j,(i2z,shift_t,shift_b) in enumerate(hrzs_z):
        n = len(i2z)

        i1s,i2s = np.zeros(n),np.zeros(n)

        if sect:
            if j == 0:
                rate = (random.uniform(shift_t[0], shift_b[0])-shift_t[0]) / (shift_b[0]-shift_t[0])
            f = interp1d([0, random.randint(20, n-20) , n-1],
                         [rate, 0.5, 1-rate], kind='linear')
            for i in range(n):
                i2s[i] = i2z[i]
                i1s[i] = shift_t[i] + f(i) * (shift_b[i]-shift_t[i])
        else:
            rate = (random.uniform(shift_t[0], shift_b[0])-shift_t[0]) / (shift_b[0]-shift_t[0])
            if big:
                rate = 0.5 + abs(rate - 0.5)
            else:
                rate = 0.5 - abs(rate - 0.5)

            for i in range(n):
                i2s[i] = i2z[i]
                i1s[i] = shift_t[i] + rate * (shift_b[i]-shift_t[i])

        hrzs_new.append((i1s,i2s))

    return hrzs_new

def map_hrzs_into_img(hrzs, size, rg=0.6):
    h, w = size
    img = np.zeros(size)
    hrzv = []
    for hrz in hrzs:
        v, c = 0, 0
        if isinstance(hrz,tuple):
            hrz = [hrz]
        for i1s,_ in hrz:
            v += i1s.sum()
            c += len(i1s)
        hrzv.append(v / c)

    for i1 in range(h):
        for i2 in range(w):
            for i,hrz in enumerate(hrzs):
                if isinstance(hrz,tuple):
                    hrz = [hrz]
                for j1s,j2s in hrz:
                    for j1,j2 in zip(j1s,j2s):
                        dist = compute_edist((i1,i2),(j1,j2))
                        if dist <= rg:
                            if i1 > j1:
                                img[i1][i2] = hrzv[i] + dist
                            else:
                                img[i1][i2] = hrzv[i] - dist
    return img/(h-1)

def map_fals_into_img(faults, size, rg=1.2):
    h, w = size
    img = np.zeros(size)
    for i1 in range(h):
        for i2 in range(w):
            for i,fault in enumerate(faults):
                j1s,j2s = fault
                for j1,j2 in zip(j1s,j2s):
                    dist = compute_edist((i1,i2),(j1,j2))
                    if dist <= rg:
                        img[i1][i2] = 1
    return img

def transform_img(img):
    u, v=img.shape

    def f(i,j):
        return i+0.1*np.sin(2*np.pi*j)
    def g(i,j):
        return j+0.1#*np.sin(3*np.pi*i)

    M , N =[], []
    for i in range(u):
        for j in range(v):
            i0, j0 = i/u, j/v
            u0=int(f(i0,j0)*300)
            v0=int(g(i0,j0)*300)
            M.append(u0)
            N.append(v0)

    m1,m2=max(M),max(N)
    n1,n2=min(M),min(N)
    oup=np.zeros((m1-n1,m2-n2))

    for i in range(u):
        for j in range(v):
            i0=i/u
            j0=j/v
            u0=int(f(i0,j0)*300)-n1-1
            v0=int(g(i0,j0)*300)-n2-1
            oup[u0,v0]=img[i,j]
    return interp_img(oup, img.shape)

def interp_img(a, size):
    h, w = size
    h0, w0 = a.shape
    y0 = np.linspace(0, h0-1, h0)
    x0 = np.linspace(0, w0-1, w0)
    f = interp2d(x0, y0, a)
    y = np.linspace(0, h0-1, h)
    x = np.linspace(0, w0-1, w)
    return f(x, y)

def remove_hrzs_near_fault(hrzs_img, fals_img):
    h, w = hrzs_img.shape
    hrzs_img_new = hrzs_img.copy()
    for i1 in range(h):
        for i2 in range(w):
            jj = np.arange(-1,1,1).tolist()
            kk = np.arange(-3,3,1).tolist()
            for j in jj:
                for k in kk:
                    ii1 = max(min(i1+j,h-1),0)
                    ii2 = max(min(i2+k,w-1),0)
                    if fals_img[ii1,ii2] == 1:
                        hrzs_img_new[i1,i2] =  0
    return hrzs_img_new

def get_ucert_rg_for_fals(fals,size,pertb=12,pertb_itv=6,stretch_en=1.2,squeeze_en=0.8,sigma=2):
    h, w = size
    fals_x,fals_z = [],[]

    for j,fal in enumerate(fals):
        i1x, i2x = fal
        fals_x.append((i1x,i2x))

        f = interp1d([i1x[i] for i in range(0,len(i1x),pertb_itv)],
                     [i2x[i] for i in range(0,len(i2x),pertb_itv)],
                     fill_value="extrapolate", kind='cubic')

        shift_t,shift_b = np.zeros(h),np.zeros(h)

        for i1 in range(h):
            if i1 < h//2:
                rate = (stretch_en-squeeze_en)*(h//2-i1)/(h//2)+squeeze_en
            else:
                rate = (stretch_en-squeeze_en)*(i1-h//2)/(h-1-h//2)+squeeze_en

            shift_t[i1] = f(i1) - pertb * rate
            shift_b[i1] = f(i1) + pertb * rate

        f_t = interp1d([i for i in range(0,h,pertb_itv)],
                       [shift_t[i] for i in range(0,h,pertb_itv)],
                     fill_value="extrapolate", kind='cubic')

        f_b = interp1d([i for i in range(0,h,pertb_itv)],
                       [shift_b[i] for i in range(0,h,pertb_itv)],
                     fill_value="extrapolate", kind='cubic')

        fals_z.append((np.linspace(0,h-1,h),
                       np.array([f_t(i) for i in range(h)]),
                       np.array([f_b(i) for i in range(h)])))

    return fals_x,fals_z

def get_ucert_rg_for_hrzs(hrzs,size,fps,pertb=12,pertb_itv=2,stretch_en=1.7,squeeze_en=0.5,sigma=6):
    h, w = size
    hrzs_x,hrzs_z = [],[]
    for j,hrz in enumerate(hrzs):
        for i,(i1s, i2s) in enumerate(hrz):
            if i == 0:
                i1x,i2x = i1s,i2s
            else:
                i1x,i2x = np.append(i1x,i1s),np.append(i2x,i2s)

        hrzs_x.append((i1x,i2x))

        f = interp1d([i2x[i] for i in range(0,len(i2x),pertb_itv)],
                     [i1x[i] for i in range(0,len(i1x),pertb_itv)],
                     fill_value="extrapolate", kind='cubic')

        shift_t,shift_b = np.zeros(w),np.zeros(w)

        f_r = interp1d([0]+sorted(fps[j])+[w-1],
                     [stretch_en]+[squeeze_en for k in range(len(fps[j]))]+[stretch_en],
                     fill_value="extrapolate", kind='linear')

        for i2 in range(w):
            rate = f_r(i2)
            shift_t[i2] = f(i2) - pertb * rate
            shift_b[i2] = f(i2) + pertb * rate

        hrzs_z.append((np.linspace(0,w-1,w),
                       gaussian_filter(shift_t, sigma=sigma),
                       gaussian_filter(shift_b, sigma=sigma)))

    return hrzs_x,hrzs_z

def find_sect_point(hrzs, fals, size):
    ilst = []
    n = size[-1]
    for hrz in hrzs:
        i1x,i2x = [],[]
        for i1s, i2s in hrz:
            i1x += i1s.tolist()
            i2x += i2s.tolist()

        f = interp1d(i2x, i1x, fill_value="extrapolate", kind='linear')
        i1x, i2x = np.zeros(n), np.zeros(n)
        for k in range(n):
            i2x[k] = k
            i1x[k] = f(i2x[k])

        rds = []
        for i1z, i2z in fals:
            mdist = 1e8
            ed = 0
            for j in range(len(i1z)):
                tmp = compute_edist((i1x,i2x), (i1z[j],i2z[j]))
                i = np.argmin(tmp)
                if tmp[i] < mdist:
                    mdist = tmp[i]
                    rd = i2x[i]
            rds.append(rd)
        ilst.append(rds)

    return ilst
