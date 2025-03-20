<p align="center">
<img src="assets/OSTrack.jpg" alt="OSTrack">
</p>

# :rocket:__OSTrack__

![result](assets/bandicam.gif)

OSTrack is an artificial intelligence technology for tracking and locking unmanned aerial vehicles based on the ViT deep network model. OSTrack is based on the Vision of Transformer deep learning model. For unmanned aerial vehicles moving at high speed in the near and far fields, it uses visual tracking to lock the position of the unmanned aerial vehicle in real-time video frames. The model mainly uses multiple initial anchor bounding boxes, obtains feature maps through feature extraction based on network input, and the position of the unmanned aerial vehicle is determined by the votes of the anchor boxes given by the network model. The trained network model has certain robustness to near and far fields, partial occlusion, and light changes.  

## 1.Environment :bulb:

| Name | Version |   Name | Version |
|------|---------|--------|---------|
| Python | 3.8.10 |   PyTorch | 2.4.1 |
| opencv-python | 4.9.0.80 |   Tkinter | 8.6 |
| pillow | 10.2.0 | torchvision | 0.19.1 |

## 2.Usage :computer:

Now you can use utils.py to get the ostrack model and use it for training and testing.the model architecture is shown below:  CEblock has 12 layers, and there is also a detection head with five layers behind it. However, in reality, the test input data of Ostrack requires template and search images. Only through manual annotation or automatic annotation can the specific location of the small - scale drone in the first - frame image of the video sequence, that is, the template image, be determined. While the search image doesn't need to be processed, and a normal sequence frame can be selected.

```bash
# if you want to use the yolo model
# you can use the following command
python app_yolo.py

# if you want to use the ostrack model
# you can use the following command
python app_osyo.py
```

| CEBlock | Detection Head |
| ------ | ------------- |
| ![1](assets/architecture/ostrack_1.jpg) | ![2](assets/architecture/ostrack_5.jpg) |

## 3.TODO :book:

- [x] Finish the model configuration code and import the vit_base_384_model_ce model.
- [x] Train an initial - frame localization model using YOLOv5 for the automatic annotation of templates.
- [x] Complete the function of drone tracking for imported videos in OSTrack.
- [x] Finish the GUI interface for OSTrack and YOLOv5 models' deployment.  

| search | template |
| ------ | ------------- |
| ![template_image](assets/uav_1.jpg) | ![search_image](assets/uav_2.jpg) |
| ![orig_video](assets/infrared_5.gif) | ![result](assets/processed_infrared_5.gif) |

## 4.YOLO Results :football:

Now this is YOLOv5 model's  training results, consisting of confusion_matrix, labels_correlogram, F1_curve, labels and PR/P/R_curve. The training results of YOLOv5 are not included in this project.  Next, you'll deploy the s/m/l/x models of YOLOv5. When you encounter the following error in a Windows 10/11 system environment: raise NotImplementedError("cannot instantiate %r on your system"), you can add the following code to the first line of the ./yolov5/utils/general.py file.

```bash
# the error is as follows:
raise NotImplementedError("cannot instantiate %r on your system")
NotImplementedError: cannot instantiate 'PosixPath' on your system

# you can add the following code to the first line of the ./yolov5/utils/general.py file.
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
```

If you want to compress drone IR video, you can choose below command to compress video to .gif format. The training dataset of Yolov5 is reconstructed based on the Got_10k drone infrared dataset used by OSTrack. The purpose of using the Yolov5 model is to locate the coordinates of the first frame image of the drone, so as to provide coordinates for the template image and search image required for tracking by the Ostrack model later.

```bash
ffmpeg -ss 00:00:05 -t 00:00:05 -i video/infrared.mp4 -vf "fps=1,scale=640:\
-1:flags=lanczos,split[s0][s1];[s0]palettegen=stats_mode=single:max_colors=16[p];\
[s1][p]paletteuse=dither=floyd_steinberg" -gifflags +transdiff -loop 0 \
-final_delay 20 -y output_3mb.gif
```

![confusion_matrix](assets/results/confusion_matrix.jpg)  

| labels | labels_correlogram |
| ------------- | ------------- |
| ![labels](assets/results/labels.jpg) | ![labels_correlogram](assets/results/labels_correlogram.jpg) |

| PR curve | P curve | R curve | F1 score |
| ------ | ------------- | ------------- | ------------- |
| ![PR_curve](assets/results/PR_curve.jpg) | ![P_curve](assets/results/P_curve.jpg) | ![R_curve](assets/results/R_curve.jpg) | ![F1_curve](assets/results/F1_curve.jpg) |

Before training, the dataset architecture is shown below. And the YOLOv5 s/m/l/x version model training results are shown below. This datasets consisting of 39965 training infrared images and 40355 valing infrared images.  The images directory are mainly about original infrared images and the labels dirrectory are mainly about the coordinates of the upper left corner of the bounding box (x, y) and the width and height of the bounding box (w, h).

| train_batch | val_batch | val preds |
| ------------- | ------------- | ------------- |
|![train_batch](assets/results/train_batch2.jpg)| ![val_batch](assets/results/val_batch2_labels.jpg)|![val_preds](assets/results/val_batch2_pred.jpg)|

![loss curve](assets/results/results.jpg)

```bash
# the yolov5 training dataset architecture is as follows.
datasets
    |
    |____images
    |       |
    |       |____train
    |       |       |____00000001.jpg
    |       |       |____00000002.jpg
    |       |       |____...
    |       |____val
    |               |____00000001.jpg
    |               |____00000002.jpg
    |               |____...
    |____labels
            |
            |____train
            |       |____00000001.txt
            |       |____00000002.txt
            |       |____...
            |____val
                    |____00000001.txt
                    |____00000002.txt
                    |____...
```

## 5.OSTrack Results :bulb:

The automatic annotation of Template Image and Search Image required by OSTrack, and then use Opencv to crop out the obtained drone center coordinates with 2 times and 5 times the size of the bounding box as Template Image and Search Image, respectively. The main purpose of using OSTrack is to eliminate the interference of infrared drone instance images by environmental background, and to reduce the positioning range of single frame images by using template and search methods to improve positioning accuracy and reduce the possibility of environmental interference, thereby greatly improving the tracking performance. The results of the OSTrack model are shown below.

```bash
# this function is used to clip frame for template and search area
def ScaleClip(img, xyxy, mode=None):
      """ScaleClip is used to clip frame for template and search area
      :param img: the frame image must consists of UAV pixels
      :param xyxy: the up-left and down-right coordinates of the UAV bounding box"""
      img_array = np.array(img)
      width, height = xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]
      center = np.array([xyxy[0] + width / 2, xyxy[1] + height / 2])
      scale_factor = {'template': 2, 'search': 5}.get(mode, 0)
      scaled_width = int(scale_factor * width)
      scaled_height = int(scale_factor * height)
      # Calculate the cropping rectangle ensuring it does not exceed image boundaries.
      top_left_x = max(int(center[0] - scaled_width / 2), 0)
      top_left_y = max(int(center[1] - scaled_height / 2), 0)
      bottom_right_x = min(int(center[0] + scaled_width / 2), img_array.shape[1])
      bottom_right_y = min(int(center[1] + scaled_height / 2), img_array.shape[0])
      # Clip the image
      img_clipped = img_array[top_left_y:bottom_right_y, top_left_x:bottom_right_x, :]
      return img_clipped
```

After testing, the ostrack pretrained model offered by the [original author](https://github.com/LY-1/MCJT) has no effect on the tracking performance. And the testing code in app_osyo.py is as follows.  

![pretrained model image](assets/ostrack_test0320.jpg)  

```bash
 # 调用OSTrack模型测试
self.xyxy = [xy.detach().cpu().numpy() for xy in self.xyxy]
# 进行OSTrack模型裁剪,调用GPUs
template_img = template_transform( ScaleClip(self.frame, self.xyxy, mode='template') ).unsqueeze(0).to(device)
search_img = search_transform( ScaleClip(self.frame, self.xyxy, mode='search') ).unsqueeze(0).to(device)
ostrack_results = self.ostrack(template_img, search_img)
ostrack_results = ostrack_results['pred_boxes'][0]
ostrack_results = ostrack_results.detach().cpu().numpy()[0]
whwh = [int(ostrack_results[0]*self.frame_width), int(ostrack_results[1]*self.frame_height),
        int(ostrack_results[2]*self.frame_width), int(ostrack_results[3]*self.frame_height)]
cv2.rectangle(self.frame, (whwh[0], whwh[1]), (whwh[2], whwh[3]), (0, 0, 255), 2)
print(f'the OSTrack results: {ostrack_results}')
print('the yolov5 preds: ',type(self.xyxy), '\t', self.xyxy)
```

When using the software test, it was found that when the drone appeared in the background to generate strong infrared light, it could not perform the normal location tracking task normally. For example, in the following two cases, there was a short-term drone tracking loss. Therefore, we consider using the YOLOv5 + OSTrack model, combined with the excellent positioning ability of the YOLOv5 model and the powerful continuous frame tracking ability of OSTrack, so as to achieve better location tracking effect.  

|   infrared uav case 1   |   infrared uav case 2   |
|   -------------------   |   -------------------   |
| ![case1](assets/exception/exception1.gif) | ![case2](assets/exception/exception2.gif) |

## 6.Thanks :heart:

```bash
# If you are interested in the original project, you can click on the link below.
https://github.com/LY-1/MCJT
```

If you need the pre-trained YOLOv5 infrared drone positioning model and OSTrack model weights, you can download it on Baidu Netdisk. The relevant download links are as follows:

```bash
# Files shared via online disk：ostrack
link: https://pan.baidu.com/s/1lPM_ACRkc-g8WDkB0tw7EA?pwd=92fk 
extraction code: 92fk
```

Meanwhile, it is declared that this project is the reproduction and improvement based on the work of [original author](https://github.com/LY-1/MCJT). We used a new self-made dataset for training and designed the GUI interface for deployment based on the trained model. Subsequently, we will also conduct actual operations to test the actual effect of the model. We are very grateful to the original author for his work. Of course, if you think our work based on this can attract you, please also give a little star.
