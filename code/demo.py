# --------------------------------------------------------
# DaSiamRPN
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
#!/usr/bin/python

import glob, cv2, torch
import numpy as np
from os.path import realpath, dirname, join

from net import SiamRPNvot
from run_SiamRPN import SiamRPN_init, SiamRPN_track
from utils import get_axis_aligned_bbox, cxy_wh_2_rect
import time

# load net
net = SiamRPNvot()
net.load_state_dict(torch.load(join(realpath(dirname(__file__)), 'SiamRPNVOT.model')))
net.eval().cuda()

# image and init box
image_files = sorted(glob.glob('./MOT16-13/*.jpg'))
#image_files = sorted(glob.glob('./bag/*.jpg'))
init_rbox = [334.02,128.36,438.19,188.78,396.39,260.83,292.23,200.41]
[cx, cy, w, h] = get_axis_aligned_bbox(init_rbox)
## MOT16-05 第一帧
#w = 69
#h = 190
#cx = 20 + w/2
#cy = 136 + h/2

## MOT16-09 第一帧
#w = 171
#h = 345
#cx = 1686 + w/2
#cy = 387 + h/2
##

## MOT16-13 第一帧
w = 52
h = 109
cx = 1521 + w/2
cy = 568 + h/2
##


# tracker init
target_pos, target_sz = np.array([cx, cy]), np.array([w, h])
print(target_pos)       ## 初始化位置
print(target_sz)        ## 初始化大小
im = cv2.imread(image_files[0])  # HxWxC
cv2.rectangle(im, (int(cx) - int(w/2), int(cy) - int(h/2)), (int(cx) + int(w/2), int(cy) + int(h/2)), (0, 255, 255), 3)
cv2.imshow('firstFrame', im)
cv2.waitKey(1)
state = SiamRPN_init(im, target_pos, target_sz, net)

time.sleep(5)
# tracking and visualization
toc = 0
for f, image_file in enumerate(image_files):
    im = cv2.imread(image_file)
    tic = cv2.getTickCount()
    ## 主要流程在此tracker中，需要细看
    state = SiamRPN_track(state, im)  # track
    toc += cv2.getTickCount()-tic
    ## 出矩形，画框，显示
    res = cxy_wh_2_rect(state['target_pos'], state['target_sz'])
    res = [int(l) for l in res]
    cv2.rectangle(im, (res[0], res[1]), (res[0] + res[2], res[1] + res[3]), (0, 255, 255), 3)
    cv2.imshow('SiamRPN', im)
    cv2.waitKey(1)

print('Tracking Speed {:.1f}fps'.format((len(image_files)-1)/(toc/cv2.getTickFrequency())))
