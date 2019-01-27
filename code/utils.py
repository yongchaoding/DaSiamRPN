# --------------------------------------------------------
# DaSiamRPN
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.cpu().numpy()
    elif type(tensor).__module__ != 'numpy':
        raise ValueError("Cannot convert {} to numpy array"
                         .format(type(tensor)))
    return tensor


def to_torch(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor"
                         .format(type(ndarray)))
    return ndarray


def im_to_numpy(img):
    img = to_numpy(img)
    img = np.transpose(img, (1, 2, 0))  # H*W*C
    return img


def im_to_torch(img):
    img = np.transpose(img, (2, 0, 1))  # C*H*W
    img = to_torch(img).float()
    return img


def torch_to_img(img):
    img = to_numpy(torch.squeeze(img, 0))
    img = np.transpose(img, (1, 2, 0))  # H*W*C
    return img

## pos 物体的位置
## model_sz 图片输入模型的大小
## original_sz 原始图片基本上两倍外拓后的大小(由于此原因需要进行补图片)

def get_subwindow_tracking(im, pos, model_sz, original_sz, avg_chans, out_mode='torch', new=False):

    if isinstance(pos, float):
        pos = [pos, pos]
    sz = original_sz
    im_sz = im.shape
    c = (original_sz+1) / 2
    context_xmin = round(pos[0] - c)  # floor(pos(2) - sz(2) / 2);
    context_xmax = context_xmin + sz - 1
    context_ymin = round(pos[1] - c)  # floor(pos(1) - sz(1) / 2);
    context_ymax = context_ymin + sz - 1
    print("context_xmin", context_xmin)
    print("context_xmax", context_xmax)
    print("context_ymin", context_ymin)
    print("context_ymax", context_ymax)

    left_pad = int(max(0., -context_xmin))
    top_pad = int(max(0., -context_ymin))
    right_pad = int(max(0., context_xmax - im_sz[1] + 1))
    bottom_pad = int(max(0., context_ymax - im_sz[0] + 1))
    print("left_pad", left_pad)
    print("top_pad", top_pad)
    print("right_pad", right_pad)
    print("bottom_pad", bottom_pad)

    context_xmin = context_xmin + left_pad
    context_xmax = context_xmax + left_pad
    context_ymin = context_ymin + top_pad
    context_ymax = context_ymax + top_pad

    print("context_xmin", context_xmin)
    print("context_xmax", context_xmax)
    print("context_ymin", context_ymin)
    print("context_ymax", context_ymax)
    print("========================")

    # zzp: a more easy speed version
    r, c, k = im.shape      ## r height c weight k channel
    ## 若有任何需求需要pad，则进入if判断，进行pad
    if any([top_pad, bottom_pad, left_pad, right_pad]):
        te_im = np.zeros((r + top_pad + bottom_pad, c + left_pad + right_pad, k), np.uint8)  # 0 is better than 1 initialization
        te_im[top_pad:top_pad + r, left_pad:left_pad + c, :] = im               ## middle image full

        ## full pad part use avg_chans
        if top_pad:
            te_im[0:top_pad, left_pad:left_pad + c, :] = avg_chans
        if bottom_pad:
            te_im[r + top_pad:, left_pad:left_pad + c, :] = avg_chans
        if left_pad:
            te_im[:, 0:left_pad, :] = avg_chans
        if right_pad:
            te_im[:, c + left_pad:, :] = avg_chans
    ## 从image中提取出exaplmer部分
        im_patch_original = te_im[int(context_ymin):int(context_ymax + 1), int(context_xmin):int(context_xmax + 1), :]
    else:
        im_patch_original = im[int(context_ymin):int(context_ymax + 1), int(context_xmin):int(context_xmax + 1), :]

    ## resize to model size
    if not np.array_equal(model_sz, original_sz):
        im_patch = cv2.resize(im_patch_original, (model_sz, model_sz))  # zzp: use cv to get a better speed
    else:
        im_patch = im_patch_original

    cv2.imshow('im_patch', im_patch)
    cv2.waitKey(1)
    return im_to_torch(im_patch) if out_mode in 'torch' else im_patch

def get_subwindow_tracking_z(im, pos, model_sz, original_sz_w, original_sz_h, out_mode='torch', new=False):

    context_xmin = round(pos[0] - original_sz_w/2)  # floor(pos(2) - sz(2) / 2);
    context_xmax = context_xmin + original_sz_w
    context_ymin = round(pos[1] - original_sz_h/2)  # floor(pos(1) - sz(1) / 2);
    context_ymax = context_ymin + original_sz_h

    im_patch_original = im[int(context_ymin):int(context_ymax), int(context_xmin):int(context_xmax), :]

    
    ## resize to model size
    im_patch = cv2.resize(im_patch_original, (model_sz, model_sz))  # zzp: use cv to get a better speed

    cv2.imshow('im_patch', im_patch)
    cv2.waitKey(1)
    return im_to_torch(im_patch) if out_mode in 'torch' else im_patch

def get_subwindow_tracking_x(im, model_sz, out_mode='torch', new=False):
    im_patch = cv2.resize(im, (model_sz, model_sz))
    cv2.imshow('im_patch', im_patch)
    cv2.waitKey(1)
    return im_to_torch(im_patch) if out_mode in 'torch' else im_patch

def cxy_wh_2_rect(pos, sz):
    return np.array([pos[0]-sz[0]/2, pos[1]-sz[1]/2, sz[0], sz[1]])  # 0-index


def rect_2_cxy_wh(rect):
    return np.array([rect[0]+rect[2]/2, rect[1]+rect[3]/2]), np.array([rect[2], rect[3]])  # 0-index


def get_axis_aligned_bbox(region):
    try:
        region = np.array([region[0][0][0], region[0][0][1], region[0][1][0], region[0][1][1],
                           region[0][2][0], region[0][2][1], region[0][3][0], region[0][3][1]])
    except:
        region = np.array(region)
    cx = np.mean(region[0::2])
    cy = np.mean(region[1::2])
    x1 = min(region[0::2])
    x2 = max(region[0::2])
    y1 = min(region[1::2])
    y2 = max(region[1::2])
    A1 = np.linalg.norm(region[0:2] - region[2:4]) * np.linalg.norm(region[2:4] - region[4:6])
    A2 = (x2 - x1) * (y2 - y1)
    s = np.sqrt(A1 / A2)
    w = s * (x2 - x1) + 1
    h = s * (y2 - y1) + 1
    return cx, cy, w, h

def show_scores(scores, fig_n):
    scores = 255 - scores * 255
    fig = plt.figure(fig_n)
    ax1 = fig.add_subplot(151)
    ax2 = fig.add_subplot(152)
    ax3 = fig.add_subplot(153)
    ax4 = fig.add_subplot(154)
    ax5 = fig.add_subplot(155)
    ax1.imshow(scores[0,:,:], interpolation='none', cmap='hot')
    ax2.imshow(scores[1,:,:], interpolation='none', cmap='hot')
    ax3.imshow(scores[2,:,:], interpolation='none', cmap='hot')
    ax4.imshow(scores[3,:,:], interpolation='none', cmap='hot')
    ax5.imshow(scores[4,:,:], interpolation='none', cmap='hot')
    plt.ion()
    plt.show()
    plt.pause(0.001)