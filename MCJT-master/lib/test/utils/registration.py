import numpy as np
import cv2

def registration(img_align, img_temp, prev_state):
    img1 = cv2.cvtColor(img_align, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img_temp, cv2.COLOR_BGR2GRAY)
    # height, width = img2.shape


    # orb_detector = cv2.ORB_create(5000)       # ORB是一种关键点检测算法，用于特征提取，5000是指最大的特征数。不指定会尽可能多地提取特征。
    orb_detector = cv2.ORB_create(2500)
    # orb_detector = cv2.AKAZE_create()
    # orb_detector = cv2.SIFT_create()

    kp1, d1 = orb_detector.detectAndCompute(img1, None)  # kp1是关键点，d1是关键点的描述符，32个8为的数组，实际是256维的二进制描述
    kp2, d2 = orb_detector.detectAndCompute(img2, None)

    d1.dtype = np.uint8
    d2.dtype = np.uint8

    # 计算每对关键点之间的描述符距离，cv2.NORM_HAMMING表示距离测量（汉明距离）
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    matches = matcher.match(d1, d2)  # 返回最佳匹配，BFMatcher.knnMatch()返回k个最佳匹配

    # matches.sorted(key = lambda x: x.distance)
    matches = sorted(matches, key=lambda x: x.distance)
    matches = matches[:int(len(matches) * 0.9)]
    # matches = matches[:int(len(matches) * 0.3)]
    no_of_matches = len(matches)
    p1 = np.zeros((no_of_matches, 2))
    p2 = np.zeros((no_of_matches, 2))
    for i in range(len(matches)):
        p1[i, :] = kp1[matches[i].queryIdx].pt  # 把匹配到的对应关键点对记录下来
        p2[i, :] = kp2[matches[i].trainIdx].pt

    p1_list = []
    p2_list = []
    for i in range(len(p1)):
        # if (p1[i, 0] <= 225 and p1[i, 1] <= 85) or (p1[i, 0] >= 430 and p1[i, 1] <= 40):   # 指定范围内的关键点不做考虑
        #     continue
        # if (p2[i, 0] <= 225 and p2[i, 1] <= 85) or (p2[i, 0] >= 430 and p2[i, 1] <= 40):
        #     continue
        if p1[i, 1] <= 85 or p2[i, 1] <= 85:
            continue
        p1_list.append([p1[i, 0], p1[i, 1]])
        p2_list.append([p2[i, 0], p2[i, 1]])
    p1_list = np.array(p1_list)
    p2_list = np.array(p2_list)

    if len(p1_list) >= 4:
        p1 = p1_list
        p2 = p2_list
    if len(p1) < 4:
        return prev_state[0], prev_state[1]

    # 为结果图像的验证创建单应性，为了得到最佳的变换，需要使用RANSAC算法检测异常值并去除
    # 创建单应性矩阵
    homography, mask = cv2.findHomography(p1, p2, cv2.RANSAC)
    if homography is None:                      # 无法找到单应性矩阵
        return prev_state[0], prev_state[1]

    point = np.array([[prev_state[0], prev_state[1]]], dtype=np.float32).reshape(-1, 1, 2)  # 二维变三维， 整形转float型， 一个都不能少
    transformed_point = cv2.perspectiveTransform(point, homography).reshape(-1, 2)
    transformed_x, transformed_y = transformed_point[0]
    return transformed_x, transformed_y