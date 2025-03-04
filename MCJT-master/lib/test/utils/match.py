import numpy as np


def iou_batch(bboxes1, bboxes2):
    """
    From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
    """
    bboxes2 = np.expand_dims(bboxes2, 0)
    bboxes1 = np.expand_dims(bboxes1, 1)

    xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
    yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    xx2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
    yy2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
              + (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1]) - wh)
    return (o)

def odis_batch_norm(bb_test, bb_gt):
    """
    odis_batch_norm(detections, trackers)，都是左上和右下点的坐标
    From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
    """
    m = bb_test.shape[0]
    n = bb_gt.shape[0]
    if m == 0 or n == 0:
        o = []
        print(o)
        return (np.array(o))
    test = np.zeros((m, 4))
    gt = np.zeros((n, 4))
    try:   # 归一化
        test[:, 0] = bb_test[:, 0] / 640
        test[:, 1] = bb_test[:, 1] / 512
        test[:, 2] = bb_test[:, 2] / 640
        test[:, 3] = bb_test[:, 3] / 512
    except:
        a=1

    gt[:, 0] = bb_gt[:, 0] / 640
    gt[:, 1] = bb_gt[:, 1] / 512
    gt[:, 2] = bb_gt[:, 2] / 640
    gt[:, 3] = bb_gt[:, 3] / 512

    o = np.zeros((m, n))
    cd = (test[:, :2] + test[:, 2:4]) / 2    # 中心点坐标
    ct = (gt[:, :2] + gt[:, 2:4]) / 2
    w = bb_gt[:, 2] - bb_gt[:, 0]
    h = bb_gt[:, 3] - bb_gt[:, 1]
    for i in range(m):
        for j in range(n):
            if w[j] == 0 or h[j] == 0 or ct[j, 0] <= 0 or ct[j, 0] >= 1 or ct[j, 1] <= 0 or ct[j, 1] >= 1:
                o[i, j] = 1
                continue
            o[i, j] = np.linalg.norm(cd[i] - ct[j])    # 根据中心点求范数，默认2范数，即欧氏距离，这里越小越好

        # min_max_scaler = preprocessing.MinMaxScaler()
        # print(o,m,n)
        # o =1 - min_max_scaler.fit_transform(o)

    o = 1 - o    # 这里越大越好

    return (o)

def size_batch_norm(bb_test, bb_gt):
    """
    odis_batch_norm(detections, trackers)，都是左上和右下点的坐标
    From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
    """
    m = bb_test.shape[0]
    n = bb_gt.shape[0]
    if m == 0 or n == 0:
        o = []
        print(o)
        return (np.array(o))

    o = np.zeros((m, n))
    prev_w = bb_test[:, 2] - bb_test[:, 0]
    prev_h = bb_test[:, 3] - bb_test[:, 1]
    w = bb_gt[:, 2] - bb_gt[:, 0]
    h = bb_gt[:, 3] - bb_gt[:, 1]
    for i in range(m):
        for j in range(n):
            if w[j] >= 0.25 * 640 or h[j] >= 0.25 * 512 or w[j] <= 0 or h[j] <= 0:
                o[i, j] = 1
                continue
            o[i, j] = abs(prev_w[i] - w[j]) / 640 + abs(prev_h[i] - h[j]) / 512    # 变化量除以图像尺寸，这里越小越好

    o = 1 - o    # 这里越大越好

    return (o)


def merge_IE_distance(iou_cost_matrix, distance_cost_matrix, size_cost_matrix):
    merged_cost_matrix = np.zeros_like(iou_cost_matrix)   # 创建一个形状和iou_cost_matrix相同的矩阵
    rows, cols = iou_cost_matrix.shape

    # param_iou = 1 / 3 * 1.0
    # param_edist = 1 / 3 * 1.0
    # param_size = 1 / 3 * 1.0

    param_iou = 0.5
    param_edist = 0.25
    param_size = 0.25

    for i in range(rows):
        for j in range(cols):
            # print("param_iou =",param_iou, "param_edist =",param_edist)
            merged_cost_matrix[i][j] = param_iou * iou_cost_matrix[i][j] + param_edist * distance_cost_matrix[i][j] + param_size * size_cost_matrix[i][j]
            # print("merged_cost_matrix[i][j] =",merged_cost_matrix[i][j])

    return merged_cost_matrix

def cal_Merge(prev_Box, preBBox):
    ious = iou_batch(prev_Box, preBBox)
    ods = odis_batch_norm(prev_Box, preBBox)
    size = size_batch_norm(prev_Box, preBBox)
    matrix = merge_IE_distance(ious, ods, size)
    return matrix

if __name__ == '__main__':
    bboxes1 = np.array([[72.35867901, 0., 84.32613566, 8.65141589]])
    bboxes2 = np.array([[70.91107389, 0., 86.43257678, 0.]])
    ious = iou_batch(bboxes1, bboxes2)
    ods = odis_batch_norm(bboxes1, bboxes2)
    size = size_batch_norm(bboxes1, bboxes2)
    matrix = merge_IE_distance(ious, ods, size)
    print(matrix)