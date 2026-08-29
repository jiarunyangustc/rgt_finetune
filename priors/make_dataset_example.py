#!/usr/bin/env python3
"""Build a horizon-subset training set while retaining segment labels.

Usage: python make_dataset_example.py <tag> <indices such as 0,2>
"""
import os, sys, numpy as np
L='../data/lunnan/'
NI,NJ,NZ=416,256,256
CENT=[0.2638,0.4061,0.7778,0.8445,0.9202]
TAG=sys.argv[1]; KEEP=[int(x) for x in sys.argv[2].split(',')]
OUT=L+f'train_input_3d_inline_{TAG}_segtile'
os.makedirs(OUT,exist_ok=True)
sx=np.fromfile(L+'ln_416x256x256.dat',np.float32).reshape(NI,NJ,NZ)
fr=np.fromfile(L+'ln_fr_wd3_x_rm_filled.dat',np.float32).reshape(NI,NJ,NZ)
seg=np.load(L+'segment_pt_faultbarrier_w3_min1000_max10000_512.npy')
def ms(a): return (a-a.mean())/(a.std()+1e-8)
nhz=[]
for i in range(NI):
    frame=fr[i].T.astype(np.float32).copy()
    keepmask=np.zeros_like(frame,bool)
    for k in KEEP:
        keepmask |= (np.abs(frame-CENT[k])<0.012)
    frame[~keepmask]=0.0
    mx=np.zeros((len(CENT),NZ,NJ),np.float32)
    for k in KEEP:
        mx[k]=(np.abs(frame-CENT[k])<1.5e-3).astype(np.float32)
    d={'seis':ms(sx[i].T).astype(np.float32)[np.newaxis,...],
       'frame':frame[np.newaxis,...],
       'mx_single':mx[np.newaxis,...],
       'mask_valid':np.ones((1,1,NZ,NJ),np.float32),
       'segments':seg[i].T[np.newaxis,...].astype(np.float32)}
    np.save(os.path.join(OUT,f'{i}.npy'),d)
    nhz.append(int((mx.sum(axis=(1,2))>0).sum()))
nhz=np.asarray(nhz)
d0=np.load(os.path.join(OUT,'0.npy'),allow_pickle=True).item()
u=np.unique(np.round(d0['frame'][d0['frame']!=0],4))
print(f'{TAG}: keep={KEEP} -> {OUT}')
print(f'  frame values {u} | horizons per section min/median '
      f'{nhz.min()}/{int(np.median(nhz))} | nonzero segment fraction '
      f'{np.mean(d0["segments"] != 0):.4f}')
