import math

from lib.models.ostrack import build_ostrack
from lib.test.tracker.basetracker import BaseTracker
import torch

from lib.test.utils.motion_constrain import getBestBox
from lib.test.tracker.vis_utils import gen_visualization
from lib.test.utils.hann import hann2d
from lib.train.data.processing_utils import sample_target
# for debug
import cv2
import os

from lib.test.tracker.data_utils import Preprocessor
from lib.utils.box_ops import clip_box
from lib.utils.ce_utils import generate_mask_cond
from lib.test.utils.cal_tracker import update_xywh, update_xyxy, update
from lib.test.utils.motion_constrain import xyxy2xywh
from lib.test.utils.registration import registration

class OSTrack(BaseTracker):
    def __init__(self, params, dataset_name):
        super(OSTrack, self).__init__(params)
        network = build_ostrack(params.cfg, training=False)
        network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu')['net'], strict=True)
        self.cfg = params.cfg
        self.network = network.cuda()
        self.network.eval()
        self.preprocessor = Preprocessor()
        self.state = None

        self.prev = None     # 用于图像配准

        self.feat_sz = self.cfg.TEST.SEARCH_SIZE // self.cfg.MODEL.BACKBONE.STRIDE
        # motion constrain
        self.output_window = hann2d(torch.tensor([self.feat_sz, self.feat_sz]).long(), centered=True).cuda()

        # for debug
        self.debug = params.debug
        self.use_visdom = params.debug
        self.frame_id = 0
        if self.debug:
            if not self.use_visdom:
                self.save_dir = "debug"
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)
            else:
                # self.add_hook()
                self._init_visdom(None, 1)
        # for save boxes from all queries
        self.save_all_boxes = params.save_all_boxes
        self.z_dict1 = {}
        self.new_z = None

    def initialize(self, image, info: dict):
        self.prev = image.copy()  # 先将模板图像赋值

        # forward the template once
        z_patch_arr, resize_factor, z_amask_arr = sample_target(image, info['init_bbox'], self.params.template_factor,
                                                    output_sz=self.params.template_size)
        self.z_patch_arr = z_patch_arr
        template = self.preprocessor.process(z_patch_arr, z_amask_arr)
        with torch.no_grad():
            self.z_dict1 = template

        self.box_mask_z = None
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            template_bbox = self.transform_bbox_to_crop(info['init_bbox'], resize_factor,
                                                        template.tensors.device).squeeze(1)
            self.box_mask_z = generate_mask_cond(self.cfg, 1, template.tensors.device, template_bbox)

        # save states
        self.state = info['init_bbox']
        self.frame_id = 0
        if self.save_all_boxes:
            '''save all predicted boxes'''
            all_boxes_save = info['init_bbox'] * self.cfg.MODEL.NUM_OBJECT_QUERIES
            return {"all_boxes": all_boxes_save}

    def generate_new_template(self, image, box):
        z_patch_arr, resize_factor, z_amask_arr = sample_target(image, box, self.params.template_factor,
                                                                output_sz=self.params.template_size)
        self.z_patch_arr = z_patch_arr
        template = self.preprocessor.process(z_patch_arr, z_amask_arr)
        with torch.no_grad():
            self.new_z = template

    def inference(self, image, rs_factor, state=None, re_search=False, new_temp_track=False):
        H, W, _ = image.shape
        self.frame_id += 1
        if state is None:
            x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.params.search_factor * rs_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)
        else:
            x_patch_arr, resize_factor, x_amask_arr = sample_target(image, state,
                                                                    self.params.search_factor * rs_factor,
                                                                    output_sz=self.params.search_size)  # (x1, y1, w, h)
        search = self.preprocessor.process(x_patch_arr, x_amask_arr)

        with torch.no_grad():
            x_dict = search
            # merge the template and the search
            # run the transformer
            if new_temp_track == False:
                out_dict = self.network.forward(
                    template=self.z_dict1.tensors, search=x_dict.tensors, ce_template_mask=self.box_mask_z)
            else:
                out_dict = self.network.forward(
                    template=self.new_z.tensors, search=x_dict.tensors, ce_template_mask=self.box_mask_z)

        # add hann windows
        pred_score_map = out_dict['score_map']
        # response = self.output_window * pred_score_map
        response = pred_score_map
        # response = pred_score_map
        pred_boxes, conf = getBestBox(response, out_dict['size_map'], out_dict['offset_map'], self.state, resize_factor, re_search)
        return pred_boxes, conf

    def track(self, image, info: dict = None):
        H, W, _ = image.shape
        resize_thre = 0.57
        resize_factor = 2
        registraion_thre = 0.1
        improve_thre = 0.57
        new_temp_track_thre = 0.2
        new_temp_improve_thre = 0.7

        pred_boxes, conf = self.inference(image, 1)
        if conf < resize_thre:
            pred_boxes2, conf2 = self.inference(image, resize_factor)  # re_search=True效果会降0.05%
            if conf < conf2:
                conf = conf2
                pred_boxes = pred_boxes2

        # 图像配准
        if conf < registraion_thre:
            x, y = registration(self.prev, image.copy(), self.state)
            if x >= 0 and y >= 0 and x <= W - self.state[2] and y <= H - self.state[3]:
                box = [x, y, self.state[2], self.state[3]]
                pred_boxes3, conf3 = self.inference(image, resize_factor, box, re_search=True)
                if conf < conf3 and improve_thre < conf3:
                    conf = conf3
                    pred_boxes = pred_boxes3

        if conf < new_temp_track_thre and self.new_z is not None:
            pred_boxes4, conf4 = self.inference(image, resize_factor, new_temp_track=True)
            if new_temp_improve_thre <= conf4:
                conf = conf4
                pred_boxes = pred_boxes4

        if conf != 0:
            self.prev = image.copy()
            self.state = pred_boxes

        # 高于阈值生成新的模板
        if conf >= 0.75:
            self.generate_new_template(image.copy(), [cor for cor in self.state])


        return {"target_bbox": self.state} if conf > 0 else {"target_bbox": [0, 0, 0, 0]}


    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1) # (N,4) --> (N,)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)

    def add_hook(self):
        conv_features, enc_attn_weights, dec_attn_weights = [], [], []

        for i in range(12):
            self.network.backbone.blocks[i].attn.register_forward_hook(
                # lambda self, input, output: enc_attn_weights.append(output[1])
                lambda self, input, output: enc_attn_weights.append(output[1])
            )

        self.enc_attn_weights = enc_attn_weights


def get_tracker_class():
    return OSTrack
