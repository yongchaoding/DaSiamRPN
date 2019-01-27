#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F


from utils import get_subwindow_tracking, show_scores
from utils import get_subwindow_tracking_z, get_subwindow_tracking_x


## 锚框的生成，此处需要仔细研究

def generate_anchor(total_stride, scales, ratios, score_size):
    anchor_num = len(ratios) * len(scales)
    print("ratios", ratios)
    print("scales", scales)
    print("anchor_num", anchor_num)

    anchor = np.zeros((anchor_num, 4),  dtype=np.float32)
    size = total_stride * total_stride
    count = 0

    ## 此处确定的是放大倍数，scale放大倍数，ratios 比例，主要指长宽比例
    for ratio in ratios:
        # ws = int(np.sqrt(size * 1.0 / ratio))
        ws = int(np.sqrt(size / ratio))
        hs = int(ws * ratio)
        for scale in scales:
            wws = ws * scale
            hhs = hs * scale
            anchor[count, 0] = 0
            anchor[count, 1] = 0
            anchor[count, 2] = wws
            anchor[count, 3] = hhs
            count += 1
    print("anchor", anchor)
    ## anchor [[   0.    0.  104.   32.]
    ## [   0.    0.   88.   40.]
    ## [   0.    0.   64.   64.]
    ## [   0.    0.   40.   80.]
    ## [   0.    0.   32.   96.]]
    ## 重复score_size * score_size 次数， 与后面的score_map表相对应
    anchor = np.tile(anchor, score_size * score_size).reshape((-1, 4))

    ## 此处生成覆盖对象，对整个score_map进行覆盖，从而可以进行卷积运算
    # print("anchor", anchor)
    ori = - (score_size / 2) * total_stride     ## 中心点位置， 后续要进行偏移
    print("ori", ori)
    xx, yy = np.meshgrid([ori + total_stride * dx for dx in range(score_size)],
                         [ori + total_stride * dy for dy in range(score_size)])
    #print("xx", xx)
    #print("yy", yy)
    xx, yy = np.tile(xx.flatten(), (anchor_num, 1)).flatten(), \
             np.tile(yy.flatten(), (anchor_num, 1)).flatten()
    #print("xx", xx)
    #print("yy", yy)
    anchor[:, 0], anchor[:, 1] = xx.astype(np.float32), yy.astype(np.float32)
    # print("anchor shape", anchor)
    return anchor


class TrackerConfig(object):
    # These are the default hyper-params for DaSiamRPN 0.3827
    windowing = 'cosine'  # to penalize large displacements [cosine/uniform]
    # Params from the network architecture, have to be consistent with the training
    #exemplar_size = 127  # input z size
    #instance_size = 271  # input x size (search region)
    exemplar_size = 127  # input z size
    instance_size = 512  # input x size (search region)
    total_stride = 8
    score_size = int(instance_size-exemplar_size)/total_stride+1
    context_amount = 0.5  # context amount for the exemplar
    ratios = [0.33, 0.5, 1, 2, 3]
    scales = [total_stride, ]                               ## 这个是否与total_stride相关呢？？
    anchor_num = len(ratios) * len(scales)
    anchor = []
    penalty_k = 0.055
    window_influence = 0 #0.42
    lr = 0.295
    # adaptive change search region #
    adaptive = True

    def update(self, cfg):
        for k, v in cfg.items():
            setattr(self, k, v)
        self.score_size = (self.instance_size - self.exemplar_size) / self.total_stride + 1
        self.scales = [self.total_stride, ]


## 还原数据， 拿出响应程度最大的回归框

def tracker_eval(net, x_crop, target_pos, target_sz, window, scale_z, p, state, s_z):
    delta, score = net(x_crop)

    delta = delta.permute(1, 2, 3, 0).contiguous().view(4, -1).data.cpu().numpy()
    score = F.softmax(score.permute(1, 2, 3, 0).contiguous().view(2, -1), dim=0).data[1, :].cpu().numpy()

    # print("score.shape", score.shape())
    ## 原始的score_map
    scoreFrame = score.reshape(p.anchor_num, p.score_size, p.score_size)
    show_scores(scoreFrame, 1)

    best_pscore_id = np.argmax(score)
    print("best delta origin:", delta[:, best_pscore_id])
    #print("------------------")
    #print("scoreFrame", scoreFrame[1, :, :])
    #print("------------------")
    
    ## 这个只是做纠正作用，不能用于真正的定位

    delta[0, :] = delta[0, :] * p.anchor[:, 2] + p.anchor[:, 0]
    delta[1, :] = delta[1, :] * p.anchor[:, 3] + p.anchor[:, 1]
    delta[2, :] = np.exp(delta[2, :]) * p.anchor[:, 2]
    delta[3, :] = np.exp(delta[3, :]) * p.anchor[:, 3]

    print("best delta:", delta[:, best_pscore_id])

    def change(r):
        return np.maximum(r, 1./r)

    def sz(w, h):
        pad = (w + h) * 0.5
        sz2 = (w + pad) * (h + pad)
        return np.sqrt(sz2)

    def sz_wh(wh):
        pad = (wh[0] + wh[1]) * 0.5
        sz2 = (wh[0] + pad) * (wh[1] + pad)
        return np.sqrt(sz2)

    # size penalty
    s_c = change(sz(delta[2, :], delta[3, :]) / (sz_wh(target_sz)))  # scale penalty
    r_c = change((target_sz[0] / target_sz[1]) / (delta[2, :] / delta[3, :]))  # ratio penalty

    penalty = np.exp(-(r_c * s_c - 1.) * p.penalty_k)
    #pscore = penalty * score
    pscore = score

    # window float
    pscore = pscore * (1 - p.window_influence) + window * p.window_influence
    best_pscore_id = np.argmax(pscore)

    ## 评判成绩时候用的score_map
    pscoreFrame = pscore.reshape(p.anchor_num, p.score_size, p.score_size)
    show_scores(pscoreFrame, 2)

    im_w = state['im_w']
    im_h = state['im_h']
    im_z = np.sqrt(im_w * im_h)

    print("s_z", s_z)

    target = delta[:, best_pscore_id] / scale_z
    target_sz = target_sz / scale_z
    lr = penalty[best_pscore_id] * score[best_pscore_id] * p.lr

    

    res_x = target[0]  + target_pos[0] #+ im_w/2  #
    res_y = target[1]   + target_pos[1] #+ im_h/2 #

    print("target[0]", target[0], "target_pos[0]", target_pos[0], "im_w", im_w)
    print("target[1]", target[1], "target_pos[1]", target_pos[1], "im_h", im_h)

    print("target[2]", target[2])
    print("target[3]", target[3])

    res_w = target_sz[0] * (1 - lr) + target[2] * lr
    res_h = target_sz[1] * (1 - lr) + target[3] * lr

    target_pos = np.array([res_x, res_y])
    target_sz = np.array([res_w, res_h])
    return target_pos, target_sz, score[best_pscore_id]


def SiamRPN_init(im, target_pos, target_sz, net):
    state = dict()
    p = TrackerConfig()
    p.update(net.cfg)

    print("p.exemplar_size", p.exemplar_size)
    print("p.instance_size", p.instance_size)
    print("p.total_stride", p.total_stride)
    print("p.score_size", p.score_size)
    print("p.context_amount", p.context_amount)
    print("p.window_influence", p.window_influence)
    print("p.lr", p.lr)

    state['im_h'] = im.shape[0]
    state['im_w'] = im.shape[1]

    ## 假设图片为正方形，则方便处理，可以用instance_size
    ## 假设图片不为正方形，是否还能使用两个不同的instance_size 进行图形的表示

    ## instance_size 调整，很重要, 我们代码中，要求搜索区域全覆盖
    if p.adaptive:
        ## 如果物体非常小， 为什么要增加搜索区域呢？  只是resize的大小
        if ((target_sz[0] * target_sz[1]) / float(state['im_h'] * state['im_w'])) < 0.004:      
            p.instance_size = 287  # small object big search region
        else:
            p.instance_size = 271   

        p.score_size = (p.instance_size - p.exemplar_size) / p.total_stride + 1

    ## 最后的score_map大小
    print("p.score_size", p.score_size)     
    ## 根据p.scales, p.ratios 生成锚框， 目前的育苗狂，似乎是位置相关，在原先的origin上进行偏移得到
    p.anchor = generate_anchor(p.total_stride, p.scales, p.ratios, int(p.score_size))
    print("p.anchor size", p.anchor.shape)      ## 目前是生成1805个锚框， 尺度上5个，平移上19*19个
    print("p.anchor", p.anchor)         

    ## 三个通道平均值
    avg_chans = np.mean(im, axis=(0, 1))
    print("avg_chans", avg_chans)

    wc_z = target_sz[0] + p.context_amount * sum(target_sz)
    print("target_sz[0]", target_sz[0], "wc_z", wc_z)
    hc_z = target_sz[1] + p.context_amount * sum(target_sz)
    print("target_sz[1]", target_sz[1], "hc_z", hc_z)
    s_z = round(np.sqrt(wc_z * hc_z))
    s_z_real = round(np.sqrt(target_sz[0] * target_sz[1]))
    print("s_z", s_z)

    # initialize the exemplar       ## 模板生成的时候为何不用原始的size，而是要用几乎扩大两倍后的东西，为防止长宽差异太大引起的问题
    z_crop = get_subwindow_tracking(im, target_pos, p.exemplar_size, s_z, avg_chans)

    ## 由于纵向压缩的太厉害了，效果没有之前那个好
    #z_crop = get_subwindow_tracking_z(im, target_pos, p.exemplar_size, target_sz[0], target_sz[1])
    ## 放入模板，进行生成
    z = Variable(z_crop.unsqueeze(0))
    net.temple(z.cuda())

    ## 过滤窗口选择，默认选择consine窗口进行过滤
    if p.windowing == 'cosine':
        window = np.outer(np.hanning(p.score_size), np.hanning(p.score_size))
    elif p.windowing == 'uniform':
        window = np.ones((p.score_size, p.score_size))
    
    window = np.tile(window.flatten(), p.anchor_num)

    state['p'] = p
    state['net'] = net
    state['avg_chans'] = avg_chans
    state['window'] = window
    state['target_pos'] = target_pos
    state['target_sz'] = target_sz
    return state


def SiamRPN_track(state, im):
    p = state['p']
    net = state['net']
    avg_chans = state['avg_chans']
    window = state['window']
    target_pos = state['target_pos']
    target_sz = state['target_sz']

    ## 一定几率的外拓
    wc_z = target_sz[1] + p.context_amount * sum(target_sz)
    hc_z = target_sz[0] + p.context_amount * sum(target_sz)
    s_z = np.sqrt(wc_z * hc_z)

    print("!!!", state['im_h']/s_z)

    scale_z = p.exemplar_size / s_z
    print("origin scale_z", scale_z)
    scale_x = state['im_w'] / p.score_size
    print("origin scale_z", scale_x)

    d_search = (p.instance_size - p.exemplar_size) / 2      ## 半边差距
    pad = d_search / scale_z                                ## 半边差距的scale，即
    s_x = s_z + 2 * pad                                     ## 外拓边缘
    print("~~~~~~~~~~~~~~~~~")
    print("scale_z", scale_z)
    print("d_search", d_search)
    print("pad", pad)
    print("s_x", s_x)
    print("~~~~~~~~~~~~~~~~~")

    # extract scaled crops for search region x at previous target position
    ## 实时的图片crop，基本与初始化的时候流程一样，只是输出图片为 instance_size

    ## 此处需要进行改进，x_crop应该是整张图片，但是之前怎么进行关联暂时还没有想好
    x_crop = Variable(get_subwindow_tracking(im, target_pos, p.instance_size, round(s_x), avg_chans).unsqueeze(0))
    #x_crop = Variable(get_subwindow_tracking_x(im, p.instance_size).unsqueeze(0))

    target_pos, target_sz, score = tracker_eval(net, x_crop.cuda(), target_pos, target_sz * scale_z, window, scale_z, p, state, s_z)
    

    ## 保证不越界
    target_pos[0] = max(0, min(state['im_w'], target_pos[0]))
    target_pos[1] = max(0, min(state['im_h'], target_pos[1]))
    target_sz[0] = max(10, min(state['im_w'], target_sz[0]))
    target_sz[1] = max(10, min(state['im_h'], target_sz[1]))
    ## 更新迭代
    state['target_pos'] = target_pos
    state['target_sz'] = target_sz
    state['score'] = score
    return state
