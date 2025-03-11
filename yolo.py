"""
TODO: 使用YOLOv5进行无人机图像检测定位测试,
      数据集采用红外影像数据./datasets/test1
      进行分割重构数据集
Time: 2025/03/07-Redal
"""
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

import os
import cv2
import argparse
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from yolov5.models.experimental import attempt_load 
from torchvision.transforms import transforms
from yolov5.utils.general import non_max_suppression, scale_boxes
from yolov5.utils.augmentations import letterbox

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
proj_path = os.getcwd()
sys.path.append(os.path.join(proj_path, "yolov5"))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
yolo_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((640, 640)),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])



##############################  配置解析文件变量  ###############################
def config():
    parser = argparse.ArgumentParser(description='YOLOv5 inference config',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument_group('YOLOv5 Dataset Reconstruction')
    parser.add_argument('--gt_dir',  type=str, default=r'datasets\images', help='ground truth file directory path(s)')
    parser.add_argument('--gt_mode', type=str, default=r'train', help='ground truth file mode [train, val]')
    parser.add_argument('--gt_file', type=str, default=r'groundtruth.txt', help='ground truth file name')
    parser.add_argument('--label_dir', type=str, default=r'datasets\labels', help='labels file directory path(s)')
    parser.add_argument('--label_mode', type=str, default=r'train', help='labels file mode [train, val]')
    parser.add_argument('--origin_dir', type=str, default=r'datasets\test', help='images file directory path(s)')
    # related to YOLOv5 model and weights
    parser.add_argument_group('YOLOv5 Model Configuration')
    parser.add_argument('--weights_dir', type=str, default=r'weights', help='weights file directory path(s)')
    parser.add_argument('--weights_file', type=str, default=r'yolov5s.pt', help='weights file name yolov5s/m/l/x.pt')
    parser.add_argument('--yolo_dir', type=str, default=r'yolov5', help='yolov5 file directory path(s)')
    args = parser.parse_args()
    return args



##############################  重构Got_10k数据集  ###############################
def split_groundtruth_to_individual_labels(args):
    """Split the label information in the groundtruth.txt file into the txt file for each image
    :param input image is 640x512 pixels width x height"""
    gt_filepath = os.path.join(args.gt_dir, args.gt_mode, args.gt_file)
    output_folder = os.path.join(args.label_dir, args.label_mode)
    img_w = 640; img_h = 512
    with open(gt_filepath, 'r') as f:
        lines = f.readlines()
    for idx,line in enumerate(lines):
        parts = line.strip().split(',')
        lr_w, lr_h = float(parts[0]), float(parts[1])
        box_w, box_h = float(parts[2]), float(parts[3])
        # normalization
        cneter_w, center_h = (lr_w + box_w / 2) / img_w , (lr_h + box_h / 2) / img_h
        bounding_w, bounding_h = box_w / img_w, box_h / img_h
        txt_file_path = os.path.join(output_folder, f'{idx:08d}.txt')
        with open(txt_file_path, 'w') as txt_file:
            txt_file.write(f'{0} {cneter_w} {center_h} {bounding_w} {bounding_h}\n')
            print(f'{txt_file_path} has been written', end='\r', flush=True)


def rebuild_data(args):
    """Rebuild the images/labels dataset form test directory"""
    img_w = 640; img_h = 512
    origin_dir = args.origin_dir
    original_dirspath = [os.path.join(origin_dir, dirname) for dirname in os.listdir(origin_dir)]
    for idx, dirpath in enumerate(original_dirspath):
        with open(os.path.join(dirpath, 'groundtruth.txt')) as gt:
            gt_lines = gt.readlines()
        if idx % 2 == 0:
            args.gt_mode = 'train'
            args.label_mode = 'train'
        else:
            args.gt_mode = 'val'
            args.label_mode = 'val'
        for id,line in enumerate(gt_lines):
            num_files = len(os.listdir(os.path.join(args.gt_dir, args.gt_mode)))
            # read image and save it to train images folder
            img_path = os.path.join(dirpath, f'{id:08d}.jpg')
            img = cv2.imread(img_path)
            img_save_path = os.path.join(args.gt_dir, args.gt_mode, f'{num_files:08d}.jpg')
            cv2.imwrite(img_save_path, img)
            # read the groundtruth.txt file into train labels folder
            parts = line.strip().split(',')
            lr_w, lr_h = float(parts[0]), float(parts[1])
            box_w, box_h = float(parts[2]), float(parts[3])
            cneter_w, center_h = (lr_w + box_w / 2) / img_w , (lr_h + box_h / 2) / img_h
            bounding_w, bounding_h = box_w / img_w, box_h / img_h
            txt_filepath = os.path.join(args.label_dir, args.label_mode, f'{num_files:08d}.txt')
            with open(txt_filepath, 'w') as txtf:
                txtf.write(f'{0} {cneter_w} {center_h} {bounding_w} {bounding_h}\n')
            print(f'{dirpath}: {txt_filepath} and {img_save_path} has been written', end='\r', flush=True)



##############################  部署Yolov5模型  ###############################
def load_yolo(args):
    """Load the yolov5 model weights and return the model
    :param args.weights_dir: weights file directory path(s)"""
    weights_fp = os.path.join(args.weights_dir, args.weights_file)
    print(f'Loading weights from {weights_fp} and yolov5 is {args.yolo_dir}')
    yolo_model = attempt_load(weights=weights_fp, device=device)
    return yolo_model.eval()
    
def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    """Plots one bounding box on image img."""
    import random
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


##############################  主函数测试分析  ###############################
if __name__ == '__main__':
    args = config()
    model = load_yolo(args).to(device)

    # 图像预处理
    img0 = cv2.imread(r'assets\uav_1.jpg')  # 原始图像
    img = letterbox(img0, new_shape=640)[0]  # 调整大小并填充
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)

    # 推理
    img = torch.from_numpy(img).to(device)
    img = img.float()  # uint8 to fp32
    img /= 255.0  # 归一化
    if img.ndimension() == 3:
        img = img.unsqueeze(0)  # 添加批次维度

    with torch.no_grad():
        pred = model(img)[0]

    # 应用NMS
    conf_thres = 0.25
    iou_thres = 0.45
    pred = non_max_suppression(pred, conf_thres, iou_thres)

    # 后处理
    for det in pred:  # 对每张图片的检测结果进行遍历
        if len(det):
            # 将预测框从img尺寸缩放回img0尺寸
            det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], img0.shape).round()

            # 打印每个检测对象的信息
            for *xyxy, conf, cls in reversed(det):
                # xyxy包含的是边界框的左上角和右下角坐标，conf是置信度，cls是类别编号
                label = f'{model.names[int(cls)]} {conf:.2f}'
                print(f"Detected object: {label} at {xyxy}")

                # 绘制边界框和标签
                plot_one_box(xyxy, img0, label=label, color=(0, 255, 0), line_thickness=3)

    # 显示处理后的图像
    cv2.imwrite('detection_result.jpg', img0)
    print("检测结果已保存为 detection_result.jpg")
    # cv2.imshow('Detection Result', img0)
    # cv2.waitKey(0)  # 等待按键关闭窗口
    # cv2.destroyAllWindows()