import os
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim

from skimage import measure
from scipy.interpolate import interp1d, interp2d, griddata
from scipy.ndimage import gaussian_filter

from PIL import Image
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.colors as mcolors
import plotly.graph_objects as go
# from plotly.subplots import make_subplots

from utils import *
from tools import *

def draw_img(img, msk=None, cmap="jet", method="bilinear",save=False,save_file=None,vmax=None,vmin=None,aspect=1):
    plt.figure(figsize=(6,6))
#     plt.imshow(img,cmap=cmap, interpolation=method,aspect='auto')
    if vmax and vmin ==None:
        plt.imshow(img,cmap=cmap, interpolation=method,aspect=aspect,vmax=vmax)
    elif vmax == None and vmin:
        plt.imshow(img,cmap=cmap, interpolation=method,aspect=aspect,vmin=vmin)
    elif vmax and vmin:
        plt.imshow(img,cmap=cmap, interpolation=method,aspect=aspect,vmax=vmax,vmin=vmin)
    else:
        plt.imshow(img,cmap=cmap,interpolation=method,aspect=aspect)
    if msk is not None:
        plt.imshow(msk, alpha=0.4, cmap='jet', interpolation='nearest',aspect=aspect)
    plt.colorbar(fraction=0.023,pad=0.02)
    if save:
        plt.savefig(save_file, dpi=100, bbox_inches='tight')
    plt.show()



def draw_slice(volume, x_slice, y_slice, z_slice, cmap='jet',clab=None):
    if len(volume.shape) > 3:
        volume = volume.squeeze()
    z, y, x = volume.shape
    cmin=np.min(volume)
    cmax=np.max(volume)

    if clab is None:
        showscale = False
    else:
        showscale = True

    # x-slice
    yy = np.arange(0, y, 1)
    zz = np.arange(0, z, 1)
    yy,zz = np.meshgrid(yy,zz)
    xx = x_slice * np.ones((y, z)).T
    vv = volume[:,:,x_slice]
    fig = go.Figure(go.Surface(
        z=zz,
        x=xx,
        y=yy,
        surfacecolor=vv,
        colorscale=cmap,
        cmin=cmin, cmax=cmax,
        showscale=showscale,
        colorbar={"title":clab,
                  "title_side":'right',
                  "len": 0.8,
                  "thickness": 8,
                  "xanchor":"right"}))

    # y-slice
    xx = np.arange(0, x, 1)
    zz = np.arange(0, z, 1)
    xx,zz = np.meshgrid(xx,zz)
    yy = y_slice * np.ones((x, z)).T
    vv = volume[:,y_slice,:]
    fig.add_trace(go.Surface(
        z=zz,
        x=xx,
        y=yy,
        surfacecolor=vv,
        colorscale=cmap,
        cmin=cmin, cmax=cmax,
        showscale=False))

    # z-slice
    xx = np.arange(0, x, 1)
    yy = np.arange(0, y, 1)
    xx,yy = np.meshgrid(xx,yy)
    zz = z_slice * np.ones((x, y)).T
    vv = volume[z_slice,:,:]
    fig.add_trace(go.Surface(
        z=zz,
        x=xx,
        y=yy,
        surfacecolor=vv,
        colorscale=cmap,
        cmin=cmin, cmax=cmax,
        showscale=False))

    fig.update_layout(
            height=400,
            width=600,
            scene = {
            "xaxis": {"nticks": 5, "title":"Corssline"},
            "yaxis": {"nticks": 5, "title":"Inline"},
            "zaxis": {"nticks": 5, "autorange":'reversed', "title":"Sample"},
            'camera_eye': {"x": 1.25, "y": 1.25, "z": 1.25},
            'camera_up': {"x": 0, "y": 0, "z": 1},
            "aspectratio": {"x": 1, "y": 1, "z": 1.05}
            },
            margin=dict(t=0, l=0, b=0))
    fig.show()


'''
def draw_samples(samples_list, attr_list, cmap=None, norm=None, save=False, save_file=None, bit=256, sample_rate=3):

    r, num = len(attr_list), len(samples_list)

    colorbar = False
    clabels = None
    methods = []
    for key in attr_list:
        if key in ["fault", "frame", "mask"]:
            methods.append("nearest")
        else:
            methods.append("bilinear")

    if cmap is None:
        cmap = []
        for key in attr_list:
            if key in ["mask", "seis"]:
                cmap.append("gray")
            else:
                cmap.append("jet")

    fig, axs = plt.subplots(r, num, sharey=True, figsize=(17*num, 10*r))
    plt.subplots_adjust(wspace=0.08, hspace=0.18)

    for j in range(r):

        attr = attr_list[j]

        if norm is not None:
            norm_ = mpl.colors.Normalize(vmin=norm[j][0], vmax=norm[j][1])
        else:
            norm_ = None

        for i in range(num):

            if attr == 'hrzs':

                pred = samples_list[i]['pred'].copy().squeeze()
                frame = samples_list[i]['frame'].copy().squeeze()
                hvs, _ = separate_hrzs(frame, pred, bit, sample_rate)
                hvs = find_near_list(hvs)
                hrzs_p = compute_out_hrzs(pred, hvs)

                colors = ['r','g','b','c','m']
                horizons = samples_list[i]['horizon_line']

                if (r == 1) & (num > 1):
                    for k, horizon in enumerate(horizons):
                        for i1s,i2s in horizon:
                            axs[i].plot(i2s, i1s, c=colors[k], linewidth=6)
                else:
                    for k, horizon in enumerate(horizons):
                        for i1s,i2s in horizon:
                            axs[j,i].plot(i2s, i1s, c=colors[k], linewidth=6)
                    for i1s,i2s in hrzs_p:
                        axs[j,i].plot(i2s, i1s, 'k--', linewidth=6)

            elif attr == 'hrzs2':

                pred = samples_list[i]['pred'].copy().squeeze()
                frame = samples_list[i]['frame'].copy().squeeze()

                seis_section = samples_list[i]['seis'].copy().squeeze()

                section = np.copy(frame)
                section[section == 0.0] = np.nan
                vmin, vmax = None, None

                hvs, hms = separate_hrzs(frame, pred, bit, sample_rate)
                hrzs_f = compute_in_hrzs(frame, hms)
                hrzs_p = compute_out_hrzs(pred, hvs)

                if (r == 1) & (num > 1):
                    for i1s,i2s in hrzs_p:
                        axs[i].plot(i2s, i1s, c='b', linewidth=4)

                elif (r > 1) & (num == 1):
                    for i1s,i2s in hrzs_p:
                        axs[j].plot(i2s, i1s, c='b', linewidth=4)

                elif (r == 1) & (num == 1):
                    for i1s,i2s in hrzs_p:
                        axs.plot(i2s, i1s, c='b', linewidth=4)

                else:
                    im = axs[j, i].imshow(seis_section, aspect='auto', cmap='gray', norm=norm_)

                    im = axs[j, i].imshow(section, aspect='auto', cmap=cmap[j],
                                          interpolation=None, norm=norm_,
                                          vmin=vmin,vmax=vmax)
                    for i1s,i2s in hrzs_p:
                        axs[j,i].plot(i2s, i1s, 'k:', linewidth=4)

            elif attr in ['frame', 'fault']:

                section = samples_list[i][attr].copy().squeeze()
                seis_section = samples_list[i]['seis'].copy().squeeze()
                vmin, vmax = 0, 1

                if attr == 'frame':
                    section[section == 0.0] = np.nan
                elif attr == 'fault':
                    section[section == 0.0] = np.nan

                if (r == 1) & (num > 1):
                    im = axs[i].imshow(seis_section, aspect='auto', cmap='gray', norm=norm_)
                    axs[i].imshow(section, aspect='auto', cmap=cmap[j], interpolation=methods[j], norm=norm_,
                                 vmin=vmin, vmax=vmax)
                elif (r > 1) & (num == 1):
                    im = axs[j].imshow(seis_section, aspect='auto', cmap='gray', norm=norm_)
                    axs[j].imshow(section, aspect='auto', cmap=cmap[j], interpolation=methods[j], norm=norm_,
                                 vmin=vmin, vmax=vmax)
                elif (r == 1) & (num == 1):
                    im = axs.imshow(seis_section, aspect='auto', cmap='gray', norm=norm_)
                    axs.imshow(section, aspect='auto', cmap=cmap[j], interpolation=methods[j], norm=norm_,
                                 vmin=vmin, vmax=vmax)
                else:
                    im = axs[j, i].imshow(seis_section, aspect='auto', cmap='gray', norm=norm_)
                    axs[j, i].imshow(section, aspect='auto', cmap=cmap[j], interpolation=methods[j], norm=norm_,
                                 vmin=vmin, vmax=vmax)

            elif attr == 'horizon_line':
                colors = ['r','g','b','c','m']
                horizons = samples_list[i][attr]
                seis_section = samples_list[i]['seis'].copy().squeeze()
                if (r == 1) & (num > 1):
                    axs[i].imshow(seis_section, aspect='auto', cmap='gray', norm=norm_)
                    for k, horizon in enumerate(horizons):
                        for i1s,i2s in horizon:
                            axs[i].plot(i2s, i1s, c=colors[k], linewidth=6)
                else:
                    axs[j,i].imshow(seis_section, aspect='auto', cmap='gray', norm=norm_)
                    for k, horizon in enumerate(horizons):
                        for i1s,i2s in horizon:
                            axs[j,i].plot(i2s, i1s, c=colors[k], linewidth=6)

            elif attr in ['cpred', 'crgt']:
                seis_section = samples_list[i]['seis'].copy().squeeze()
                if attr == 'crgt':
                    section = samples_list[i]['rgt'].copy().squeeze()
                elif attr == 'cpred':
                    section = samples_list[i]['pred'].copy().squeeze()

                if (r == 1) & (num > 1):
                    axs[i].contour(section,np.linspace(np.min(section),np.max(section),20),\
                                  cmap='jet',linewidths=2)

                elif (r > 1) & (num == 1):
                    axs[j].contour(section,np.linspace(np.min(section),np.max(section),20),\
                                  cmap='jet',linewidths=2)

                elif (r == 1) & (num == 1):
                    axs.contour(section,np.linspace(np.min(section),np.max(section),20),\
                                  cmap='jet',linewidths=2)

                else:
                    im = axs[j, i].imshow(seis_section, aspect='auto', cmap='gray', norm=norm_)
                    axs[j,i].contour(section,np.linspace(np.min(section),np.max(section),20),\
                                  cmap='jet',linewidths=2)

            else:

                section = samples_list[i][attr].copy().squeeze()

                if (r == 1) & (num > 1):
                    im = axs[i].imshow(section, aspect='auto', cmap=cmap[j], interpolation=methods[j], norm=norm_)
                elif (r > 1) & (num == 1):
                    im = axs[j].imshow(section, aspect='auto', cmap=cmap[j], interpolation=methods[j], norm=norm_)
                elif (r == 1) & (num == 1):
                    im = axs.imshow(section, aspect='auto', cmap=cmap[j], interpolation=methods[j], norm=norm_)
                else:
                    im = axs[j, i].imshow(section, aspect='auto', cmap=cmap[j], interpolation=methods[j], norm=norm_)

            if (r == 1) & (num > 1):
                axs[i].set_xlabel('X', fontsize=36)
                axs[i].set_ylabel('Y', fontsize=36)
                axs[i].tick_params(labelsize=36)
                if clabels is not None:
                    axs[i].set_title(f'{clabels[i]}', fontsize=36)
                if colorbar:
                    fig.colorbar(im, ax=axs[i], pad=0.02)
            elif (r > 1) & (num == 1):
                axs[j].set_xlabel('X', fontsize=36)
                axs[j].set_ylabel('Y', fontsize=36)
                axs[j].tick_params(labelsize=36)
                if clabels is not None:
                    axs[j].set_title(f'{clabels[i]}', fontsize=36)
                if colorbar:
                    fig.colorbar(im, ax=axs[i], pad=0.02)
            elif (r == 1) & (num == 1):
                axs.set_xlabel('X', fontsize=36)
                axs.set_ylabel('Y', fontsize=36)
                axs.tick_params(labelsize=36)
                if clabels is not None:
                    axs.set_title(f'{clabels[i]}', fontsize=36)
                if colorbar:
                    fig.colorbar(im, ax=axs, pad=0.02)
            else:
                axs[j, i].set_xlabel('X', fontsize=36)
                axs[j, i].set_ylabel('Y', fontsize=36)
                axs[j, i].tick_params(labelsize=36)
                axs[j, i].set_xticks([])
                axs[j, i].set_yticks([])
                if clabels is not None:
                    axs[j, i].set_title(f'{clabels[i]}', fontsize=36)
                if colorbar:
                    fig.colorbar(im, ax=axs[j, i], pad=0.02)
    if save:
        plt.savefig(save_file, dpi=100, bbox_inches='tight')
    plt.show()
'''
def draw_samples(samples_list, attr_list, cmap=None, norm=None, save=False, save_file=None, bit=256, sample_rate=3, fl_max=0.05, wspace=0.08, hspace=0.18, seis_max=None, seis_min=None, laterl=False):
    r, num = len(attr_list), len(samples_list)

    if laterl:
        r, num = len(samples_list), len(attr_list)

    colorbar = False
    clabels = None
    methods = []
    for key in attr_list:
        if key in ["fault", 'fault_p', "frame", "frame_part", "maskfr", "maskfl", 'mask', 'maskloss', 'pred_fl', 'frame_all', 'cigfacies']:
            methods.append("nearest")
        else:
            methods.append("bilinear")

    if cmap is None:
        cmap = []
        for key in attr_list:
            if key in ["maskfr", "maskfl", "seis", 'mask', 'maskloss', 'seis_m', 'mask_seis', 'cigfacies']:
                cmap.append("gray")
            else:
                cmap.append("jet")

    fig, axs = plt.subplots(r, num, sharey=True, figsize=(17*num, 10*r))
    plt.subplots_adjust(wspace=wspace, hspace=hspace)

    for j in range(r):
        attr = attr_list[j]

        if norm is not None:
            norm_ = mcolors.Normalize(vmin=norm[j][0], vmax=norm[j][1])
        else:
            norm_ = None

        for i in range(num):
            if attr == 'hrzs':
                pred = samples_list[i]['pred'].copy().squeeze()
                frame = samples_list[i]['frame'].copy().squeeze()
                hvs, _ = separate_hrzs(frame, pred, bit, sample_rate)
                hvs = find_near_list(hvs)
                hrzs_p = compute_out_hrzs(pred, hvs)

                colors = ['r', 'g', 'b', 'c', 'm']
                horizons = samples_list[i]['horizon_line']

                if (r == 1) & (num > 1):
                    for k, horizon in enumerate(horizons):
                        for i1s, i2s in horizon:
                            axs[i].plot(i2s, i1s, c=colors[k], linewidth=6)
                else:
                    for k, horizon in enumerate(horizons):
                        for i1s, i2s in horizon:
                            axs[j, i].plot(i2s, i1s, c=colors[k], linewidth=6)
                    for i1s, i2s in hrzs_p:
                        axs[j, i].plot(i2s, i1s, 'k--', linewidth=6)

            elif attr == 'hrzs2':
                pred = samples_list[i]['pred'].copy().squeeze()
                frame = samples_list[i]['frame'].copy().squeeze()
                seis_section = samples_list[i]['seis'].copy().squeeze()
                section = np.copy(frame)
                section[section == 0.0] = np.nan
                vmin, vmax = 0, 1
                section_norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
                hvs, hms = separate_hrzs(frame, pred, bit, sample_rate)
                hrzs_f = compute_in_hrzs(frame, hms)
                hrzs_p = compute_out_hrzs(pred, hvs)

                if (r == 1) & (num > 1):
                    im = axs[i].imshow(seis_section, aspect='auto', cmap='gray', norm=norm_, interpolation="bilinear", vmax=seis_max, vmin=seis_min)
                    im = axs[i].imshow(section, aspect='auto', cmap=cmap[j], interpolation=None, norm=section_norm)
                    for i1s, i2s in hrzs_p:
                        axs[i].plot(i2s, i1s, 'r:', linewidth=8)

                elif (r > 1) & (num == 1):
                    im = axs[j].imshow(seis_section, aspect='auto', cmap='gray', norm=norm_, interpolation="bilinear", vmax=seis_max, vmin=seis_min)
                    im = axs[j].imshow(section, aspect='auto', cmap=cmap[j], interpolation=None, norm=section_norm)
                    for i1s, i2s in hrzs_p:
                        axs[j].plot(i2s, i1s, 'r:', linewidth=8)

                elif (r == 1) & (num == 1):
                    im = axs.imshow(seis_section, aspect='auto', cmap='gray', norm=norm_, interpolation="bilinear", vmax=seis_max, vmin=seis_min)
                    im = axs.imshow(section, aspect='auto', cmap=cmap[j], interpolation=None, norm=section_norm)
                    for i1s, i2s in hrzs_p:
                        axs.plot(i2s, i1s, 'r:', linewidth=8)

                else:
                    im = axs[j, i].imshow(seis_section, aspect='auto', cmap='gray', norm=norm_, interpolation="bilinear", vmax=seis_max, vmin=seis_min)
                    im = axs[j, i].imshow(section, aspect='auto', cmap=cmap[j], interpolation=None, norm=section_norm)
                    for i1s, i2s in hrzs_p:
                        axs[j, i].plot(i2s, i1s, 'r:', linewidth=8)

            elif attr == 'hrzs3':
                pred = samples_list[i]['pred'].copy().squeeze()
                frame = samples_list[i]['frame'].copy().squeeze()
                frame2 = samples_list[i]['fr_slope'].copy().squeeze()
                seis_section = samples_list[i]['seis'].copy().squeeze()
                section = np.copy(frame)
                section[section == 0.0] = np.nan
                section2 = np.copy(frame2)
                section2[section2 == 0.0] = np.nan
                vmin, vmax = 0, 1
                section_norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
                hvs, hms = separate_hrzs(frame, pred, bit, sample_rate)
                hrzs_f = compute_in_hrzs(frame, hms)
                hrzs_p = compute_out_hrzs(pred, hvs)
                hrzs_slope = []
                for k in range(frame2.shape[0]):
                    x = np.linspace(0, frame2.shape[1]-1, frame2.shape[1])
                    y = frame2[k]
                    hrzs_slope.append([y, x])

                if (r == 1) & (num > 1):
                    im = axs[i].imshow(seis_section, aspect='auto', cmap='gray', norm=norm_, interpolation="bilinear", vmax=seis_max, vmin=seis_min)
                    im = axs[i].imshow(section, aspect='auto', cmap=cmap[j], interpolation=None, norm=section_norm)
                    for i1s, i2s in hrzs_p:
                        axs[i].plot(i2s, i1s, 'r:', linewidth=8)
                    for i1s, i2s in hrzs_slope:
                        axs[i].plot(i2s, i1s, 'b--', linewidth=8)

                elif (r > 1) & (num == 1):
                    im = axs[j].imshow(seis_section, aspect='auto', cmap='gray', norm=norm_, interpolation="bilinear", vmax=seis_max, vmin=seis_min)
                    im = axs[j].imshow(section, aspect='auto', cmap=cmap[j], interpolation=None, norm=section_norm)
                    for i1s, i2s in hrzs_p:
                        axs[j].plot(i2s, i1s, 'r:', linewidth=8)
                    for i1s, i2s in hrzs_slope:
                        axs[j].plot(i2s, i1s, 'b--', linewidth=8)

                elif (r == 1) & (num == 1):
                    im = axs.imshow(seis_section, aspect='auto', cmap='gray', norm=norm_, interpolation="bilinear", vmax=seis_max, vmin=seis_min)
                    im = axs.imshow(section, aspect='auto', cmap=cmap[j], interpolation=None, norm=section_norm)
                    for i1s, i2s in hrzs_p:
                        axs.plot(i2s, i1s, 'r:', linewidth=8)
                    for i1s, i2s in hrzs_slope:
                        axs.plot(i2s, i1s, 'b--', linewidth=8)
                else:
                    im = axs[j, i].imshow(seis_section, aspect='auto', cmap='gray', norm=norm_, interpolation="bilinear", vmax=seis_max, vmin=seis_min)
                    im = axs[j, i].imshow(section, aspect='auto', cmap=cmap[j], interpolation=None, norm=section_norm)
                    for i1s, i2s in hrzs_p:
                        axs[j, i].plot(i2s, i1s, 'r:', linewidth=8)
                    for i1s, i2s in hrzs_slope:
                        axs[j, i].plot(i2s, i1s, 'b--', linewidth=8)

            elif attr in ['frame', 'fault', 'frame_part', 'fault_p']:
                section = samples_list[i][attr].copy().squeeze()
                seis_section = samples_list[i]['seis'].copy().squeeze()
                vmin, vmax = 0, 1
                section_norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
                if attr == 'frame' or attr == 'frame_part':
                    section[section == 0.0] = np.nan
                elif attr == 'fault' or attr == 'fault_p':
                    section[section == 0.0] = np.nan

                if (r == 1) & (num > 1):
                    im = axs[i].imshow(seis_section, aspect='auto', cmap='gray', norm=norm_, interpolation="bilinear", vmax=seis_max, vmin=seis_min)
                    axs[i].imshow(section, aspect='auto', cmap=cmap[j], interpolation=methods[j], norm=section_norm)
                elif (r > 1) & (num == 1):
                    im = axs[j].imshow(seis_section, aspect='auto', cmap='gray', norm=norm_, interpolation="bilinear", vmax=seis_max, vmin=seis_min)
                    axs[j].imshow(section, aspect='auto', cmap=cmap[j], interpolation=methods[j], norm=section_norm)
                elif (r == 1) & (num == 1):
                    im = axs.imshow(seis_section, aspect='auto', cmap='gray', norm=norm_, interpolation="bilinear", vmax=seis_max, vmin=seis_min)
                    axs.imshow(section, aspect='auto', cmap=cmap[j], interpolation=methods[j], norm=section_norm)
                else:
                    im = axs[j, i].imshow(seis_section, aspect='auto', cmap='gray', norm=norm_, interpolation="bilinear", vmax=seis_max, vmin=seis_min)
                    axs[j, i].imshow(section, aspect='auto', cmap=cmap[j], interpolation=methods[j], norm=section_norm)

            elif attr in ['pred_fl']:
                section = samples_list[i][attr].copy().squeeze()
                seis_section = samples_list[i]['seis'].copy().squeeze()
                vmin, vmax = 0, fl_max
                section[section < vmax] = np.nan

                if (r == 1) & (num > 1):
                    im = axs[i].imshow(seis_section, aspect='auto', cmap='gray', norm=norm_, interpolation="bilinear", vmax=seis_max, vmin=seis_min)
                    axs[i].imshow(section, aspect='auto', cmap=cmap[j], interpolation=methods[j], vmin=vmin, vmax=vmax)
                elif (r > 1) & (num == 1):
                    im = axs[j].imshow(seis_section, aspect='auto', cmap='gray', norm=norm_, interpolation="bilinear", vmax=seis_max, vmin=seis_min)
                    axs[j].imshow(section, aspect='auto', cmap=cmap[j], interpolation=methods[j], vmin=vmin, vmax=vmax)
                elif (r == 1) & (num == 1):
                    im = axs.imshow(seis_section, aspect='auto', cmap='gray', norm=norm_, interpolation="bilinear", vmax=seis_max, vmin=seis_min)
                    axs.imshow(section, aspect='auto', cmap=cmap[j], interpolation=methods[j], vmin=vmin, vmax=vmax)
                else:
                    im = axs[j, i].imshow(seis_section, aspect='auto', cmap='gray', norm=norm_, interpolation="bilinear", vmax=seis_max, vmin=seis_min)
                    axs[j, i].imshow(section, aspect='auto', cmap=cmap[j], interpolation=methods[j], vmin=vmin, vmax=vmax)

            elif attr == 'horizon_line':
                colors = ['r', 'g', 'b', 'c', 'm']
                horizons = samples_list[i][attr]
                seis_section = samples_list[i]['seis'].copy().squeeze()
                if (r == 1) & (num > 1):
                    axs[i].imshow(seis_section, aspect='auto', cmap='gray', norm=norm_, vmax=seis_max, vmin=seis_min)
                    for k, horizon in enumerate(horizons):
                        for i1s, i2s in horizon:
                            axs[i].plot(i2s, i1s, c=colors[k], linewidth=6)
                else:
                    axs[j, i].imshow(seis_section, aspect='auto', cmap='gray', norm=norm_, vmax=seis_max, vmin=seis_min)
                    for k, horizon in enumerate(horizons):
                        for i1s, i2s in horizon:
                            axs[j, i].plot(i2s, i1s, c=colors[k], linewidth=6)

            elif attr in ['cpred', 'crgt']:
                seis_section = samples_list[i]['seis'].copy().squeeze()
                if attr == 'crgt':
                    section = samples_list[i]['rgt'].copy().squeeze()
                elif attr == 'cpred':
                    section = samples_list[i]['pred'].copy().squeeze()

                if (r == 1) & (num > 1):
                    im = axs[i].imshow(seis_section, aspect='auto', cmap='gray', norm=norm_, interpolation="bilinear", vmax=seis_max, vmin=seis_min)
                    axs[i].contour(section, np.linspace(np.min(section), np.max(section), 20), cmap='jet', linewidths=8)

                elif (r > 1) & (num == 1):
                    im = axs[j].imshow(seis_section, aspect='auto', cmap='gray', norm=norm_, interpolation="bilinear", vmax=seis_max, vmin=seis_min)
                    axs[j].contour(section, np.linspace(np.min(section), np.max(section), 20), cmap='jet', linewidths=8)

                elif (r == 1) & (num == 1):
                    im = axs.imshow(seis_section, aspect='auto', cmap='gray', norm=norm_, interpolation="bilinear", vmax=seis_max, vmin=seis_min)
                    axs.contour(section, np.linspace(np.min(section), np.max(section), 20), cmap='jet', linewidths=8)

                else:
                    im = axs[j, i].imshow(seis_section, aspect='auto', cmap='gray', norm=norm_, interpolation="bilinear", vmax=seis_max, vmin=seis_min)
                    axs[j, i].contour(section, np.linspace(np.min(section), np.max(section), 20), cmap='jet', linewidths=8)

            elif attr in ['crgt_fault']:
                seis_section = samples_list[i]['seis'].copy().squeeze()
                section1 = samples_list[i]['rgt'].copy().squeeze()
                section2 = samples_list[i]['fault'].copy().squeeze()
                vmin, vmax = 0, 1
                section2[section2 == 0.0] = np.nan
                section_norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
                if (r == 1) & (num > 1):
                    axs[i].contour(section1, np.linspace(np.min(section1), np.max(section1), 20), cmap='jet', linewidths=2)

                elif (r > 1) & (num == 1):
                    axs[j].contour(section1, np.linspace(np.min(section1), np.max(section1), 20), cmap='jet', linewidths=2)

                elif (r == 1) & (num == 1):
                    im = axs.imshow(seis_section, aspect='auto', cmap='gray', norm=norm_, interpolation="bilinear", vmax=seis_max, vmin=seis_min)
                    axs.contour(section1, np.linspace(np.min(section1), np.max(section1), 20), cmap='jet', linewidths=2)
                    axs.imshow(section2, aspect='auto', cmap='jet', interpolation='nearest', norm=section_norm)
                else:
                    im = axs[j, i].imshow(seis_section, aspect='auto', cmap='gray', norm=norm_, interpolation="bilinear", vmax=seis_max, vmin=seis_min)
                    axs[j, i].contour(section1, np.linspace(np.min(section1), np.max(section1), 20), cmap='jet', linewidths=2)



            elif attr in ['pred', 'cigfacies',"segments"]:
                vmin, vmax = 0, 1
                section = samples_list[i][attr].copy().squeeze()


                section_norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

                if (r == 1) & (num > 1):
                    im = axs[i].imshow(section, aspect='auto', cmap=cmap[j], interpolation=methods[j])
                elif (r > 1) & (num == 1):
                    im = axs[j].imshow(section, aspect='auto', cmap=cmap[j], interpolation=methods[j])
                elif (r == 1) & (num == 1):
                    im = axs.imshow(section, aspect='auto', cmap=cmap[j], interpolation=methods[j])
                else:
                    im = axs[j, i].imshow(section, aspect='auto', cmap=cmap[j], interpolation=methods[j])
            else:

                section = samples_list[i][attr].copy().squeeze()

                if (r == 1) & (num > 1):
                    im = axs[i].imshow(section, aspect='auto', cmap=cmap[j], interpolation=methods[j], norm=norm_,vmax = seis_max,vmin = seis_min)
                elif (r > 1) & (num == 1):
                    im = axs[j].imshow(section, aspect='auto', cmap=cmap[j], interpolation=methods[j], norm=norm_,vmax = seis_max,vmin = seis_min)
                elif (r == 1) & (num == 1):
                    im = axs.imshow(section, aspect='auto', cmap=cmap[j], interpolation=methods[j], norm=norm_,vmax = seis_max,vmin = seis_min)
                else:
                    im = axs[j, i].imshow(section, aspect='auto', cmap=cmap[j], interpolation=methods[j], norm=norm_,vmax = seis_max,vmin = seis_min)

            if (r == 1) & (num > 1):
#                axs[j].set_xlabel('X', fontsize=36)
#                axs[j].set_ylabel('Y', fontsize=36)
#                axs[j].tick_params(labelsize=36)
                axs[j].set_xticks([])
                axs[j].set_yticks([])
                if clabels is not None:
                    axs[i].set_title(f'{clabels[i]}', fontsize=36)
                if colorbar:
                    fig.colorbar(im, ax=axs[i], pad=0.02)
            elif (r > 1) & (num == 1):
#                axs[j].set_xlabel('X', fontsize=36)
#                axs[j].set_ylabel('Y', fontsize=36)
#                axs[j].tick_params(labelsize=36)
                axs[j].set_xticks([])
                axs[j].set_yticks([])
                if clabels is not None:
                    axs[j].set_title(f'{clabels[i]}', fontsize=36)
                if colorbar:
                    fig.colorbar(im, ax=axs[i], pad=0.02)
            elif (r == 1) & (num == 1):
#                axs.set_xlabel('X', fontsize=36)
#                axs.set_ylabel('Y', fontsize=36)
#                axs.tick_params(labelsize=36)
                axs.set_xticks([])
                axs.set_yticks([])
                if clabels is not None:
                    axs.set_title(f'{clabels[i]}', fontsize=36)
                if colorbar:
                    fig.colorbar(im, ax=axs, pad=0.02)
            else:
#                 axs[j, i].set_xlabel('X', fontsize=36)
#                 axs[j, i].set_ylabel('Y', fontsize=36)
                axs[j, i].tick_params(labelsize=36)
                axs[j, i].set_xticks([])
                axs[j, i].set_yticks([])
                if clabels is not None:
                    axs[j, i].set_title(f'{clabels[i]}', fontsize=36)
                if colorbar:
                    fig.colorbar(im, ax=axs[j, i], pad=0.02)
    if save:
        plt.savefig(save_file, dpi=100, bbox_inches='tight')
    plt.show()

def draw_samples_3d(samples_list, attr_list,
                    slice_axis=0, slice_idx=None,
                    cmap=None, norm=None,
                    save=False, save_file=None,
                    wspace=0.08, hspace=0.18,
                    seis_max=None, seis_min=None):
    """
    Visualize one slice of a 3-D array.
    slice_axis: 0-D, 1-H, 2-W
    slice_idx: integer or None; None selects the central slice
    """
    r, num = len(attr_list), len(samples_list)
    colorbar = False
    clabels = None
    methods = []
    for key in attr_list:
        if key in ["fault", 'fault_p', "frame", "frame_part", "maskfr", "maskfl", 'mask', 'maskloss', 'pred_fl', 'frame_all', 'cigfacies']:
            methods.append("nearest")
        else:
            methods.append("bilinear")
    if cmap is None:
        cmap = []
        for key in attr_list:
            if key in ["maskfr", "maskfl", "seis", 'mask', 'maskloss', 'seis_m', 'mask_seis', 'cigfacies']:
                cmap.append("gray")
            else:
                cmap.append("jet")

    fig, axs = plt.subplots(r, num, sharey=True, figsize=(6*num, 5*r))
    plt.subplots_adjust(wspace=wspace, hspace=hspace)

    for j in range(r):
        attr = attr_list[j]
        if norm is not None:
            norm_ = mcolors.Normalize(vmin=norm[j][0], vmax=norm[j][1])
        else:
            norm_ = None

        for i in range(num):
            data = samples_list[i][attr].copy().squeeze()

            if slice_idx is None:
                idx = data.shape[slice_axis] // 2
            else:
                idx = slice_idx
            if slice_axis == 0:
                section = data[idx, :, :]
            elif slice_axis == 1:
                section = data[:, idx, :]
            elif slice_axis == 2:
                section = data[:, :, idx]
            else:
                raise ValueError("slice_axis must be 0, 1, or 2")

            # seismic section
            if 'seis' in samples_list[i]:
                seis_data = samples_list[i]['seis'].copy().squeeze()
                if slice_axis == 0:
                    seis_section = seis_data[idx, :, :]
                elif slice_axis == 1:
                    seis_section = seis_data[:, idx, :]
                elif slice_axis == 2:
                    seis_section = seis_data[:, :, idx]
            else:
                seis_section = None


            if r == 1 and num > 1:
                ax = axs[i]
            elif r > 1 and num == 1:
                ax = axs[j]
            elif r == 1 and num == 1:
                ax = axs
            else:
                ax = axs[j, i]

            if seis_section is not None:
                ax.imshow(seis_section, aspect='auto', cmap='gray', norm=norm_, interpolation="bilinear", vmax=seis_max, vmin=seis_min)
                im = ax.imshow(section, aspect='auto', cmap=cmap[j], interpolation=methods[j], norm=norm_, alpha=0.7)
            else:
                im = ax.imshow(section, aspect='auto', cmap=cmap[j], interpolation=methods[j], norm=norm_)

            ax.set_xticks([])
            ax.set_yticks([])
            if clabels is not None:
                ax.set_title(f'{clabels[i]}', fontsize=16)
            if colorbar:
                fig.colorbar(im, ax=ax, pad=0.02)

    if save:
        plt.savefig(save_file, dpi=100, bbox_inches='tight')
    plt.show()

def plot_seis_rgt_3d(seis, rgt, slice_axis=0, slice_idx=None, cmap_seis='gray', cmap_rgt='jet'):
    """
    Visualize corresponding slices from 3-D seismic and RGT arrays.
    seis, rgt: 3D numpy arrays
    slice_axis: 0, 1, or 2 selects the slicing direction
    slice_idx: slice index; None selects the center
    """
    assert seis.shape == rgt.shape
    nz, ny, nx = seis.shape

    if slice_idx is None:
        idx = [seis.shape[slice_axis] // 2]
    elif isinstance(slice_idx, int):
        idx = [slice_idx]
    else:
        idx = slice_idx

    fig = go.Figure()


    for i in idx:
        if slice_axis == 0:
            section = seis[i, :, :]
            x, y = np.arange(nx), np.arange(ny)
            xx, yy = np.meshgrid(x, y)
            zz = np.ones_like(xx) * i
            fig.add_trace(go.Surface(z=zz, x=xx, y=yy, surfacecolor=section, colorscale=cmap_seis, showscale=True, opacity=0.7, name='seis'))
        elif slice_axis == 1:
            section = seis[:, i, :]
            x, z = np.arange(nx), np.arange(nz)
            xx, zz = np.meshgrid(x, z)
            yy = np.ones_like(xx) * i
            fig.add_trace(go.Surface(z=zz, x=xx, y=yy, surfacecolor=section, colorscale=cmap_seis, showscale=True, opacity=0.7, name='seis'))
        elif slice_axis == 2:
            section = seis[:, :, i]
            y, z = np.arange(ny), np.arange(nz)
            yy, zz = np.meshgrid(y, z)
            xx = np.ones_like(yy) * i
            fig.add_trace(go.Surface(z=zz, x=xx, y=yy, surfacecolor=section, colorscale=cmap_seis, showscale=True, opacity=0.7, name='seis'))


    for i in idx:
        if slice_axis == 0:
            section = rgt[i, :, :]
            x, y = np.arange(nx), np.arange(ny)
            xx, yy = np.meshgrid(x, y)
            zz = np.ones_like(xx) * i + 0.2
            fig.add_trace(go.Surface(z=zz, x=xx, y=yy, surfacecolor=section, colorscale=cmap_rgt, showscale=True, opacity=0.5, name='rgt'))
        elif slice_axis == 1:
            section = rgt[:, i, :]
            x, z = np.arange(nx), np.arange(nz)
            xx, zz = np.meshgrid(x, z)
            yy = np.ones_like(xx) * i + 0.2
            fig.add_trace(go.Surface(z=zz, x=xx, y=yy, surfacecolor=section, colorscale=cmap_rgt, showscale=True, opacity=0.5, name='rgt'))
        elif slice_axis == 2:
            section = rgt[:, :, i]
            y, z = np.arange(ny), np.arange(nz)
            yy, zz = np.meshgrid(y, z)
            xx = np.ones_like(yy) * i + 0.2
            fig.add_trace(go.Surface(z=zz, x=xx, y=yy, surfacecolor=section, colorscale=cmap_rgt, showscale=True, opacity=0.5, name='rgt'))

    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'
        ),
        width=900,
        height=700,
        margin=dict(t=0, l=0, b=0)
    )
    fig.show()

import plotly.graph_objects as go
import numpy as np

def plot_cube_volume(data, cmap='gray', opacity=0.1, surface_count=15, clim=None, name='cube'):
    """
    Render an entire 3-D volume.
    data: 3D numpy array
    cmap: color map
    opacity: volume opacity
    surface_count: number of isosurfaces
    clim: optional (vmin, vmax)
    name: legend label
    """
    if clim is None:
        vmin, vmax = np.nanmin(data), np.nanmax(data)
    else:
        vmin, vmax = clim

    fig = go.Figure(data=go.Volume(
        x=np.arange(data.shape[2]).repeat(data.shape[0]*data.shape[1]),
        y=np.tile(np.arange(data.shape[1]).repeat(data.shape[0]), data.shape[2]),
        z=np.tile(np.arange(data.shape[0]), data.shape[1]*data.shape[2]),
        value=data.flatten(order='C'),
        opacity=opacity,
        surface_count=surface_count,
        colorscale=cmap,
        cmin=vmin,
        cmax=vmax,
        name=name,
        showscale=True
    ))

    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'
        ),
        width=900,
        height=700,
        margin=dict(t=0, l=0, b=0)
    )
    fig.show()


# plot_cube_volume(seis, cmap='gray', opacity=0.1, surface_count=15, name='seis')
# plot_cube_volume(rgt, cmap='jet', opacity=0.1, surface_count=15, name='rgt')

def plot_cube_slices(data,
                     x_idx=None, y_idx=None, z_idx=None,
                     cmap='gray', clim=None, name='cube',
                     data_2=None, cmap2='jet', clim2=None, opacity2=0.5):
    """
    Display three orthogonal slices, optionally with a second volume overlay.
    data: 3D numpy array
    data_2: 3D numpy array or None
    x_idx, y_idx, z_idx: slice indices; None selects the center
    cmap, cmap2: color maps
    clim, clim2: optional value ranges
    opacity2: opacity of data_2
    name: legend label
    """
    z, y, x = data.shape
    if x_idx is None:
        x_idx = 0
    if y_idx is None:
        y_idx = 0
    if z_idx is None:
        z_idx = z-1
    if clim is None:
        vmin, vmax = np.nanmin(data), np.nanmax(data)
    else:
        vmin, vmax = clim

    fig = go.Figure()

    # x-slice (crossline)
    yy = np.arange(0, y, 1)
    zz = np.arange(0, z, 1)
    yy, zz = np.meshgrid(yy, zz)
    xx = x_idx * np.ones_like(yy)
    vv = data[:, :, x_idx]
    fig.add_trace(go.Surface(
        z=zz, x=xx, y=yy, surfacecolor=vv,
        colorscale=cmap, cmin=vmin, cmax=vmax,
        showscale=True, name=f'{name}-x', opacity=0.9
    ))

    if data_2 is not None:
        vv2 = data_2[:, :, x_idx]
        mask2 = (vv2 != 0).astype(float)
        if clim2 is None:
            vmin2, vmax2 = np.nanmin(vv2[mask2==1]), np.nanmax(vv2[mask2==1])
        else:
            vmin2, vmax2 = clim2
        fig.add_trace(go.Surface(
            z=zz, x=xx, y=yy, surfacecolor=vv2,
            colorscale=cmap2, cmin=vmin2, cmax=vmax2,
            showscale=False, name=f'{name}_2-x',
            opacity=opacity2,
            opacityscale=[[0, 0], [1e-6, 0], [1, 1]],
        ))

    # y-slice (inline)
    xx = np.arange(0, x, 1)
    zz = np.arange(0, z, 1)
    xx, zz = np.meshgrid(xx, zz)
    yy = y_idx * np.ones_like(xx)
    vv = data[:, y_idx, :]
    fig.add_trace(go.Surface(
        z=zz, x=xx, y=yy, surfacecolor=vv,
        colorscale=cmap, cmin=vmin, cmax=vmax,
        showscale=False, name=f'{name}-y', opacity=0.9
    ))
    if data_2 is not None:
        vv2 = data_2[:, y_idx, :]
        mask2 = (vv2 != 0).astype(float)
        if clim2 is None:
            vmin2, vmax2 = np.nanmin(vv2[mask2==1]), np.nanmax(vv2[mask2==1])
        else:
            vmin2, vmax2 = clim2
        fig.add_trace(go.Surface(
            z=zz, x=xx, y=yy, surfacecolor=vv2,
            colorscale=cmap2, cmin=vmin2, cmax=vmax2,
            showscale=False, name=f'{name}_2-y',
            opacity=opacity2,
            opacityscale=[[0, 0], [1e-6, 0], [1, 1]],
        ))

    # z-slice (time/depth)
    xx = np.arange(0, x, 1)
    yy = np.arange(0, y, 1)
    xx, yy = np.meshgrid(xx, yy)
    zz = z_idx * np.ones_like(xx)
    vv = data[z_idx, :, :]
    fig.add_trace(go.Surface(
        z=zz, x=xx, y=yy, surfacecolor=vv,
        colorscale=cmap, cmin=vmin, cmax=vmax,
        showscale=False, name=f'{name}-z', opacity=0.9
    ))
    if data_2 is not None:
        vv2 = data_2[z_idx, :, :]
        mask2 = (vv2 != 0).astype(float)
        if clim2 is None:
            vmin2, vmax2 = np.nanmin(vv2[mask2==1]), np.nanmax(vv2[mask2==1])
        else:
            vmin2, vmax2 = clim2
        fig.add_trace(go.Surface(
            z=zz, x=xx, y=yy, surfacecolor=vv2,
            colorscale=cmap2, cmin=vmin2, cmax=vmax2,
            showscale=False, name=f'{name}_2-z',
            opacity=opacity2,
            opacityscale=[[0, 0], [1e-6, 0], [1, 1]]
        ))

    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data',
            zaxis=dict(autorange='reversed')
        ),
        width=800,
        height=600,
        margin=dict(t=0, l=0, b=0)
    )
    fig.show()
