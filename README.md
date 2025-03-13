# __OSTrack__

![result](assets/bandicam.gif)

OSTrack is an artificial intelligence technology for tracking and locking unmanned aerial vehicles based on the ViT deep network model. OSTrack is based on the Vision of Transformer deep learning model. For unmanned aerial vehicles moving at high speed in the near and far fields, it uses visual tracking to lock the position of the unmanned aerial vehicle in real-time video frames. The model mainly uses multiple initial anchor bounding boxes, obtains feature maps through feature extraction based on network input, and the position of the unmanned aerial vehicle is determined by the votes of the anchor boxes given by the network model. The trained network model has certain robustness to near and far fields, partial occlusion, and light changes.  

## 1.Environment

| Name | Version |   Name | Version |
|------|---------|--------|---------|
| Python | 3.8.10 |   PyTorch | 2.4.1 |
| opencv-python | 4.9.0.80 |   Tkinter | 8.6 |
| pillow | 10.2.0 | torchvision | 0.19.1 |

## 2.Usage

Now you can use utils.py to get the ostrack model and use it for training and testing.the model architecture is shown below:  CEblock has 12 layers, and there is also a detection head with five layers behind it. However, in reality, the test input data of Ostrack requires template and search images. Only through manual annotation or automatic annotation can the specific location of the small - scale drone in the first - frame image of the video sequence, that is, the template image, be determined. While the search image doesn't need to be processed, and a normal sequence frame can be selected.

| CEBlock | Detection Head |
| ------ | ------------- |
| ![1](assets/architecture/ostrack_1.jpg) | ![2](assets/architecture/ostrack_5.jpg) |

## 3.TODO

- [x] Finish the model configuration code and import the vit_base_384_model_ce model.
- [x] Train an initial - frame localization model using YOLOv5 for the automatic annotation of templates.
- [x] Complete the function of drone tracking for imported videos in OSTrack.
- [x] Finish the GUI interface for OSTrack and YOLOv5 models' deployment.  

| search | template |
| ------ | ------------- |
| ![template_image](assets/uav_1.jpg) | ![search_image](assets/uav_2.jpg) |
| ![orig_video](assets/infrared_5.gif) | ![result](assets/processed_infrared_5.gif) |

## 4.YOLO Results

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
|![train_batch](assets/results/train_batch0.jpg)| ![val_batch](assets/results/val_batch0_labels.jpg)|![val_preds](assets/results/val_batch0_pred.jpg)|

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

## 5.OSTrack Results

The automatic annotation of Template Image and Search Image required by OSTrack, and then use Opencv to crop out the obtained drone center coordinates with 2 times and 5 times the size of the bounding box as Template Image and Search Image, respectively.

## 6.Thanks

```bash
# If you are interested in the original project, you can click on the link below.
https://github.com/LY-1/MCJT
```

Meanwhile, it is declared that this project is the reproduction and improvement based on the work of [original author](https://github.com/LY-1/MCJT). We used a new self-made dataset for training and designed the GUI interface for deployment based on the trained model. Subsequently, we will also conduct actual operations to test the actual effect of the model. We are very grateful to the original author for his work. Of course, if you think our work based on this can attract you, please also give a little star.
