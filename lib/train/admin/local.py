class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/media/estar/Data/LY/Models/Anti-UAV/Ostrack-New/tracking'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = '/media/estar/Data/LY/Models/Anti-UAV/Ostrack-New/tracking/tensorboard'    # Directory for tensorboard files.
        self.pretrained_networks = '/media/estar/Data/LY/Models/Anti-UAV/Ostrack-New/tracking/pretrained_networks'
        self.lasot_dir = '/media/estar/Data/LY/Anti-UAV/img/lasot'
        self.got10k_dir = '/media/estar/Data/LY/Anti-UAV/img/got10k/train'
        self.got10k_val_dir = '/media/estar/Data/LY/Anti-UAV/img/got10k/val'
        self.lasot_lmdb_dir = '/media/estar/Data/LY/Anti-UAV/img/lasot_lmdb'
        self.got10k_lmdb_dir = '/media/estar/Data/LY/Anti-UAV/img/got10k_lmdb'
        self.trackingnet_dir = '/media/estar/Data/LY/Anti-UAV/img/trackingnet'
        self.trackingnet_lmdb_dir = '/media/estar/Data/LY/Anti-UAV/img/trackingnet_lmdb'
        self.coco_dir = '/media/estar/Data/LY/Anti-UAV/img/coco'
        self.coco_lmdb_dir = '/media/estar/Data/LY/Anti-UAV/img/coco_lmdb'
        self.lvis_dir = ''
        self.sbd_dir = ''
        self.imagenet_dir = '/media/estar/Data/LY/Anti-UAV/img/vid'
        self.imagenet_lmdb_dir = '/media/estar/Data/LY/Anti-UAV/img/vid_lmdb'
        self.imagenetdet_dir = ''
        self.ecssd_dir = ''
        self.hkuis_dir = ''
        self.msra10k_dir = ''
        self.davis_dir = ''
        self.youtubevos_dir = ''
