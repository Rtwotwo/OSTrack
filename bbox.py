"""
TODO: 构建yolov5模型的解码输出,包括bbox的解码和置信度的解码
      以及无人机的位置pixels信息
时间: 2025/03/11-Redal
"""
import os
import cv2
import random
import torch
import numpy as np
from yolov5.utils.general import non_max_suppression, scale_boxes
from yolov5.utils.augmentations import letterbox
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def plot_one_box(x, img, color=None, label=None, line_thickness=None):
      """Plots one bounding box on image img"""
      tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
      color = color or [random.randint(0, 255) for _ in range(3)]
      c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
      cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
      if label:
            tf = max(tl - 1, 1)
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)
            cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
      return img


def decoder(model, img0):
      """decode the yolo model output and plot the bounding box
      :param model: the trained yolo model consisting of s/m/l/x version
      :param img0: the uav frame image got from computer camera"""
      img = letterbox(img0, new_shape=640)[0]
      img = img[:, :, ::-1].transpose(2, 0, 1)
      img = np.ascontiguousarray(img)
      # yolo model inference and postprocess
      img = torch.from_numpy(img).to(device)
      img = img.float() / 255.0 
      if img.ndimension() == 3:
            img = img.unsqueeze(0)
      with torch.no_grad():
            pred = model(img)[0]
      # use NMS to remove redundant boxes
      conf_thres = 0.25
      iou_thres = 0.45
      pred = non_max_suppression(pred, conf_thres, iou_thres)
      for det in pred: 
            if len(det):
                  det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], img0.shape).round()
                  for *xyxy, conf, cls in reversed(det):
                  # Xyxy contains the coordinates of the upper left and lower right corners
                  #  of the bounding box, conf is the confidence level, and cls is the category number.
                        label = f'{model.names[int(cls)]} {conf:.2f}'
                        print(f"Detected object: {label} at {xyxy}")
                        plot_one_box(xyxy, img0, label=label, color=(0, 255, 0), line_thickness=3)


