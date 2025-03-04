"""
任务: 创建基本的OSTrack模型所需的实例已经辅助函数,
      同时包括相关的机器视觉处理函数
时间: 2025/01/13-Redal
"""
import os
import cv2
import yaml
import torch
import numpy as np
import importlib
from models.ostrack.ostrack import build_ostrack
from models.ostrack.vit_ce import vit_base_patch16_224_ce

def ComputeHistogramImage(frame, hist_height=200, hist_width=300):
    """使用 OpenCV 绘制 RGB 直方图
    :param frame: 输入帧(BGR 格式)
    :param hist_height: 直方图图像的高度
    :param hist_width: 直方图图像的宽度
    :return: 直方图图像
    """
    # 创建灰色背景图像
    hist_image = np.full((hist_height, hist_width, 3), fill_value=0, dtype=np.uint8)
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  

    # 计算每个通道的直方图
    for i, color in enumerate(colors):
        hist = cv2.calcHist([frame], [i], None, [256], [0, 256])
        cv2.normalize(hist, hist, 0, hist_height - 20, cv2.NORM_MINMAX)  # 留出空间给坐标轴
        for j in range(1, 256):
            # 约束直方图在图像范围内
            x1 = (j - 1) * (hist_width // 256)
            y1 = hist_height - 10 - int(hist[j - 1])  # 留出空间给横轴
            x2 = j * (hist_width // 256)
            y2 = hist_height - 10 - int(hist[j])  # 留出空间给横轴
            cv2.line(hist_image, (x1, y1), (x2, y2), color, thickness=2)
    cv2.line(hist_image, (0, hist_height - 10), (hist_width, hist_height - 10), (0, 0, 0), thickness=2)  # 横轴
    cv2.line(hist_image, (10, 0), (10, hist_height - 10), (0, 0, 0), thickness=2)  

    # 添加横轴刻度线和标签
    for i in range(0, 256, 32): 
        x = i * (hist_width // 256)
        cv2.line(hist_image, (x, hist_height - 10), (x, hist_height - 5), (255, 255, 255), thickness=1) 
        cv2.putText(hist_image, str(i), (x - 10, hist_height - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1) 
    # 添加纵轴刻度线和标签
    for i in range(0, hist_height - 10, 50): 
        y = hist_height - 10 - i
        cv2.line(hist_image, (10, y), (15, y), (255, 255, 255), thickness=1) 
        cv2.putText(hist_image, str(i), (20, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1) 
    return hist_image


def CalculateSpectrogramImage(frame, spec_height=200, spec_width=200):
    """使用 OpenCV 绘制频谱图
    :param frame: 输入帧(BGR 格式)
    :param spec_height: 频谱图图像的高度
    :param spec_width: 频谱图图像的宽度
    :return: 频谱图图像
    """
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fft = np.fft.fft2(gray_frame)
    fft_shift = np.fft.fftshift(fft)  # 将低频部分移到中心
    magnitude_spectrum = np.log(np.abs(fft_shift) + 1)  # 计算幅度谱并取对数

    magnitude_spectrum = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    spectrogram_image = cv2.resize(magnitude_spectrum, (spec_width, spec_height))

    # 频谱图美化:应用伪彩色映射
    spectrogram_color = cv2.applyColorMap(spectrogram_image, cv2.COLORMAP_JET)
    grid_color = (255, 255, 255) 
    grid_spacing = 50 
    for x in range(0, spec_width, grid_spacing):
        cv2.line(spectrogram_color, (x, 0), (x, spec_height), grid_color, 1)
    for y in range(0, spec_height, grid_spacing):
        cv2.line(spectrogram_color, (0, y), (spec_width, y), grid_color, 1)
    # 添加标题和坐标轴标签
    title = "Spectrogram"
    cv2.putText(spectrogram_color, title, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(spectrogram_color, "Frequency", (10, spec_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    cv2.putText(spectrogram_color, "Time", (spec_width - 50, spec_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    # 调整对比度和亮度
    alpha = 1.2  # 对比度系数
    beta = 30    # 亮度系数
    spectrogram_color = cv2.convertScaleAbs(spectrogram_color, alpha=alpha, beta=beta)
    return spectrogram_color


###############################  配置模型的解析文件  ############################
def load_config(config_path):
    """read the configuration file"""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config 


###############################  主函数测试分析  ################################ 
if __name__ == '__main__':
    config_path = './config/vitb_384_mae_ce_32x4_got10k_ep100.yaml'
    config = load_config(config_path)
    model = build_ostrack(config, training=False)
    print(model)