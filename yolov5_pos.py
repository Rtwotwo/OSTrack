"""
TODO: 使用YOLOv5进行无人机图像检测定位测试,
      数据集采用红外影像数据./datasets/test1
Time: 2025/03/07-Redal
"""
import os
import cv2
import argparse
import sys
import torch
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



##############################  配置解析文件变量  ###############################
def config():
    parser = argparse.ArgumentParser(description='YOLOv5 inference config',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--gt_dir',  type=str, default=r'datasets\images', help='ground truth file directory path(s)')
    parser.add_argument('--gt_mode', type=str, default=r'train', help='ground truth file mode [train, val]')
    parser.add_argument('--gt_file', type=str, default=r'groundtruth.txt', help='ground truth file name')
    parser.add_argument('--label_dir', type=str, default=r'datasets\labels', help='labels file directory path(s)')
    parser.add_argument('--label_mode', type=str, default=r'train', help='labels file mode [train, val]')
    parser.add_argument('--origin_dir', type=str, default=r'datasets\test', help='images file directory path(s)')
    args = parser.parse_args()
    return args


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
                


##############################  主函数测试分析  ###############################
if __name__ == '__main__':
    args = config()
    # split_groundtruth_to_individual_labels(args)
    rebuild_data(args)