"""
任务: 导入模型,创建GUI界面,实现对实时视频
      捕获或者视频导入的无人机定位
时间: 2025/01/13-Redal
"""
import os
import sys
import cv2 
import threading
import torch
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from util import config
from util import ComputeHistogramImage
from util import CalculateSpectrogramImage
from torchvision.transforms import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_path)
template_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((192, 192)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
search_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((384, 384)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])



########################  定义GUI界面类  #######################
class OSTrackGUI(tk.Frame):
    """设计OSTrack主界面,用于完成多功能的介绍
       以及相关功能的选择使用"""
    def __init__(self, root=None):
        super().__init__()
        self.root = root
        self.__set_widgets()
        self.frame = None
        self.video_cap = cv2.VideoCapture(0)

        self.is_running = False
        self.video_cap = None
        self.video_thread = None
        self.live_video_flag = False
        self.import_video_flag = False
        # 初始化模型
        self.ostrack = config()
        self.template_transform = template_transform
        self.sreach_transform = search_transform

    def __set_widgets(self):
        self.root.title("OSTrack GUI-Redal")
        self.root.geometry("800x600")
        self.video_label = tk.Label(self.root, text="视频显示区域", width=500, height=400); self.video_label.place(x=0, y=0)
        self.title_label = tk.Label(self.root, text='无人机目标追踪', font=("仿宋", 15), fg="black", width=30, height=2); self.title_label.place(x=505, y=0)
        self.histogram_label = tk.Label(self.root, text="直方图显示区域", font=("仿宋", 10), fg="black", width=300, height=200); self.histogram_label.place(x=0, y=400)
        self.spectrogram_label = tk.Label(self.root, text='频谱图显示区域', font=("仿宋", 10), fg="black", width=200, height=200); self.spectrogram_label.place(x=300, y=400)
        self.message_title_label  = tk.Label(self.root, text='主要功能信息提示', font=("仿宋", 15), fg="black", width=30, height=2 ); self.message_title_label.place(x=505, y=400)
        self.message_label = tk.Label(self.root,text="很感激您能使用我们的软件\n请选择您需要的功能......",
                            font=("仿宋", 12),width=40,height=5, wraplength=300, justify="left"); self.message_label.place(x=505, y=440)

        # 软件主要功能的选择按钮,按钮大小(80, 30)
        self.live_video_button = tk.Button(self.root, text='实时视频', height=1, width=10, command=self.__start_live_video__); self.live_video_button.place(x=550, y=50)
        self.import_video_button = tk.Button(self.root, text='导入视频', height=1, width=10, command=self.__start_import_video__); self.import_video_button.place(x=550, y=80)
        self.exit_button = tk.Button(self.root, text='退出程序', height=1, width=10, command=self.root.quit); self.exit_button.place(x=550, y=110)
        self.live_video_change_button = tk.Button(self.root, text='退出实时', height=1, width=10, command=self.__end_live_video__); self.live_video_change_button.place(x=670, y=50)
        self.import_video_change_button = tk.Button(self.root, text='退出导入', height=1, width=10, command=self.__end_import_video__); self.import_video_change_button.place(x=670, y=80)
        self.frame_shot_button = tk.Button(self.root, text='截取图像', height=1, width=10, command=self.__frame_shot__); self.frame_shot_button.place(x=670, y=110)

        # 界面初始显示图像
        self.main_window_img = cv2.resize( cv2.imread("images/main_window_img.png"), (500, 400) )
        self.main_window_img = ImageTk.PhotoImage(image = Image.fromarray(self.main_window_img))
        self.video_label.config(image=self.main_window_img)
        self.video_label.image = self.main_window_img

        # 应用模型进行跟踪处理按钮
        self.track_model_button = tk.Button(self.root, text='视频跟踪', height=1, width=10, command=self.__track_model__); self.track_model_button.place(x=550, y=170)
        self.video_export_button = tk.Button(self.root, text='视频导出', height=1, width=10, command=self.__video_export__); self.video_export_button.place(x=670, y=170)

    def __start_live_video__(self):
        """功能: 实时视频捕获"""
        if self.video_cap is not None:
            self.video_cap.release()
        self.live_video_flag = True
        self.import_video_flag = False
        text = "实时视频: 调用电脑或外置设备相机\n进行实时视频捕捉,再进行视频跟踪。\n注意: 退出此模式,请双击'退出实时'\n实时视频捕获中......"
        self.message_label.config(text=text)

        self.video_cap = cv2.VideoCapture(0) # 使用默认电脑相机
        self.is_running = True
        self.video_thread = threading.Thread(target=self.__video_loop__)
        self.video_thread.daemon = True
        self.video_thread.start()
    
    def __end_live_video__(self):
        """功能: 退出实时视频捕获"""
        self.is_running = False
        if self.video_cap is not None:
            self.video_cap.release()
        self.live_video_flag = False
        self.video_cap = None
        self.video_thread = None
        # 恢复为初始状态
        self.video_label.config(image=self.main_window_img)
        self.video_label.image = self.main_window_img

    def __start_import_video__(self):
        """功能: 导入用户视频"""
        file_path = filedialog.askopenfilename(filetypes=[("视频文件", "*.mp4 *.avi *.mov")])
        if file_path:
            if self.video_cap is not None:
                self.video_cap.release()
            self.import_video_flag = True
            self.live_video_flag = False
            text = "导入视频: 导入用户自定义视频文件\n进行视频帧捕捉,再进行跟踪处理。\n注意: 退出此模式,请双击'退出导入'\n用户视频导入中......"
            self.message_label.config(text=text)

            self.video_cap = cv2.VideoCapture(file_path)  # 打开视频文件
            self.is_running = True
            self.video_thread = threading.Thread(target=self.__video_loop__)
            self.video_thread.daemon = True
            self.video_thread.start()
    
    def __end_import_video__(self):
        """功能: 退出导入视频捕获"""
        self.is_running = False
        if self.video_cap is not None:
            self.video_cap.release()
        self.import_video_flag = False
        self.video_cap = None
        self.video_thread = None
        # 恢复为初始状态
        self.video_label.config(image=self.main_window_img)
        self.video_label.image = self.main_window_img

    def __frame_shot__(self):
        """功能: 截取当前视频帧"""
        frame_save_dir = 'frames'
        if not os.path.exists(frame_save_dir):
            os.makedirs(frame_save_dir) 
        frame_number = len(os.listdir(frame_save_dir))
        img_rgb = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        cv2.imwrite(f"frames/shot_{frame_number}.jpg", img_rgb)
        text = "截取图像: 用户自定义截取主页面图像\n截取图像已存放在frames文件夹。\n图像截取中......"
        self.message_label.config(text=text)

    def __track_model__(self):
        """功能: 应用模型进行跟踪处理"""
        text = "视频跟踪: 对视频进行无人机跟踪\n并将无人机以边框的形势展示。\n无人机视频跟踪中......"
        self.message_label.config(text=text)
        pass

    def __video_export__(self):
        """功能: 导出当前处理好的用户自定义的视频"""
        text = "视频导出: 对用户导入视频跟踪处理\n处理视频存放在processed文件夹。\n无人机视频导出中......"
        self.message_label.config(text=text)
        pass

    def __video_loop__(self):
        """程序主界面播放视频"""
        while self.is_running and self.video_cap.isOpened():
            success, frame = self.video_cap.read()
            if success:
                if self.live_video_flag:
                    self.frame = cv2.flip(cv2.resize( cv2.cvtColor(frame, 
                                cv2.COLOR_BGR2RGB), (500, 400)) ,1)
                else: self.frame = cv2.resize( cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (500, 400))
                # 计算直方图以及频谱图并显示
                hist_image = ComputeHistogramImage(cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB))
                spec_image = CalculateSpectrogramImage(self.frame)
                self.__show_frame__(hist_image=hist_image, spec_image=spec_image)
            else:break   
            self.root.update_idletasks() 

    def __show_frame__(self, hist_image=None, spec_image=None):
        """固定在self.video_label上显示视频"""
        img_fromarray  = Image.fromarray(self.frame)
        imgtk = ImageTk.PhotoImage(image=img_fromarray)
        self.video_label.config(image=imgtk)
        self.video_label.image = imgtk
        # 绘制直方图和频谱图
        if hist_image is not None:
            hist_image = ImageTk.PhotoImage(image=Image.fromarray(hist_image))
            self.histogram_label.config(image=hist_image)
            self.histogram_label.image = hist_image
        if spec_image is not None:
            spec_image = ImageTk.PhotoImage(image=Image.fromarray(spec_image))
            self.spectrogram_label.config(image=spec_image)
            self.spectrogram_label.image = spec_image
        self.root.after(20)



################################  主控测试函数  #############################
if __name__=='__main__':
    root = tk.Tk()
    app = OSTrackGUI(root=root)
    app.mainloop()