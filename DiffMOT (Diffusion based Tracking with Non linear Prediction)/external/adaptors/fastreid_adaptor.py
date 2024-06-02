import torch
import sys
import cv2
sys.path.append("D:/Diffusion_based_MOT/DiffMOT_main/external")
from YOLOX.yolox.data.data_augment import preproc 
from fast_reid.fastreid.config import get_cfg
from fast_reid.fastreid.modeling.meta_arch import build_model
from fast_reid.fastreid.utils.checkpoint import Checkpointer


def setup_cfg(config_file, opts):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(config_file)
    cfg.merge_from_list(opts)
    cfg.MODEL.BACKBONE.PRETRAIN = False
    cfg.freeze()
    return cfg


class FastReID(torch.nn.Module):
    def __init__(self, weights_path):
        super().__init__()
        config_file = "D:/Diffusion_based_MOT/DiffMOT_main/external/fast_reid/configs/MOT17/sbs_S50.yml"
        self.cfg = setup_cfg(config_file, ['MODEL.WEIGHTS', weights_path])
        self.model = build_model(self.cfg)
        self.model.eval()
        frame = cv2.imread("D:/Diffusion_based_MOT/custom_model_output/det_root/1.png")
        preproc_frame,_ = preproc(frame,(720,1280))
        frame = torch.from_numpy(preproc_frame).unsqueeze(0)
        frame = frame.float()
        a = self.model(frame)
        # print("a : ",a)
        # self.model.cuda()

        Checkpointer(self.model).load(weights_path)
        self.pH, self.pW = self.cfg.INPUT.SIZE_TEST

    def forward(self, batch):
        # Uses half during training
        # batch = batch.half()
        with torch.no_grad():
            return self.model(batch)
