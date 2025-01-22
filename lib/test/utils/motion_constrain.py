import torch
import numpy as np
from lib.test.utils.match import cal_Merge
from lib.utils.box_ops import clip_box
from lib.test.utils.cal_tracker import update_xywh, update_xyxy, update
from collections import deque


def getBox(offset_map, size_map, i, j, resize_factor):             # 根据特征点的位置得到框的参数
    box = [(j + offset_map[0, i, j]) / 24 * 384 / resize_factor,   # j才是和x对应的
           (i + offset_map[1, i, j]) / 24 * 384 / resize_factor,
            size_map[0, i, j] * 384 / resize_factor,
            size_map[1, i, j] * 384 / resize_factor
            ]
    return box

def map_box_back(prev_state, pred_box: list, resize_factor: float):   # 将裁剪区域上的框映射为原图，返回xyxy
    cx_prev, cy_prev = prev_state[0] + 0.5 * prev_state[2], prev_state[1] + 0.5 * prev_state[3]
    cx, cy, w, h = pred_box
    half_side = 0.5 * 384 / resize_factor      # 384是搜索范围大小
    cx_real = cx + (cx_prev - half_side)
    cy_real = cy + (cy_prev - half_side)
    return [cx_real - 0.5 * w, cy_real - 0.5 * h, cx_real + 0.5 * w, cy_real + 0.5 * h]

def xyxy2xywh(box):    # xyxy -> xywh
    x1, y1, x2, y2 = box
    return [x1, y1, x2 - x1, y2 - y1]

def xywh2xyxy(box):    # xyxy -> xywh
    x1, y1, w, h = box
    return [x1, y1, x1 + w, y1 + h]


def getBestBox(score_map, size_map, offset_map, prev_state, resize_factor, re_search=False):
    score_map = torch.squeeze(score_map).cpu().numpy()
    offset_map = torch.squeeze(offset_map).cpu().numpy()
    size_map = torch.squeeze(size_map).cpu().numpy()

    # 存置信度
    cs = []

    # 存框参数
    Boxes = []
    # score_thre = 0.018      # 低置信度阈值
    score_thre = 0.1      # 低置信度阈值
    # score_thre = 0.1      # 低分阈值


    cost_thre = 0.15
    # cost_thre = np.finfo(float).min
    # cs_thre = 0

    # 将置信度大于阈值的框的参数存下来
    m, n = score_map.shape
    for i in range(m):
        for j in range(n):
            score = score_map[i, j]
            if score >= score_thre:
                box = map_box_back(prev_state, getBox(offset_map, size_map, i, j, resize_factor), resize_factor)
                Boxes.append(box)
                cs.append(score)

    Boxes = np.array(Boxes)
    ini_Box = [0, 0, 0, 0]
    if len(Boxes) == 0:          # 最大置信度小于0.1，认为没有目标
        return ini_Box, 0

    if re_search:      # 如果是重搜索，则不计算代价矩阵
        cs = np.array(cs)
        max_index = cs.argmax()
        conf = cs[max_index]
        res = clip_box(xyxy2xywh(Boxes[max_index]), 512, 640, margin=10)
        return res, conf

    prev_state = np.array([prev_state])
    prev_state[:, 2:] = prev_state[:, :2] + prev_state[:, 2:]

    merged_cost_matrix = cal_Merge(prev_state, Boxes)  # 得到代价矩阵

    max_index = getMaxIndex(merged_cost_matrix, cs, cost_thre)
    if max_index != -1:        # 认为预测结果可靠
        res = clip_box(xyxy2xywh(Boxes[max_index]), 512, 640, margin=10)
        conf = cs[max_index]
        return res, conf
    return ini_Box, 0


def getMaxIndex(merged_cost_matrix, cs, cost_thre):
    cs = np.array([cs])
    mask = (merged_cost_matrix >= cost_thre)
    max_index = np.where(mask, cs, np.finfo(float).min).argmax()
    if cs[0, max_index] != np.finfo(float).min:
        return max_index
    return -1

# def getMaxIndex(merged_cost_matrix, cs, cost_thre, cs_thre, flag):
#     cs = np.array([cs])
#     if flag:
#         return cs.argmax()
#     if merged_cost_matrix.max() < cost_thre or cs.max() < cs_thre:
#         return -1
#     mask = (merged_cost_matrix >= cost_thre)
#     mask2 = (cs >= cs_thre)
#     temp = np.where(mask, cs, np.finfo(float).min)
#     max_index = np.where(mask2, temp, np.finfo(float).min).argmax()
#     if cs[0, max_index] != np.finfo(float).min:
#         return max_index
#     return -1