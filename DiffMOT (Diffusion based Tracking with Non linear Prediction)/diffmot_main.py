
import cv2
import torch
import argparse
from external.YOLOX.yolox.models.build import yolox_custom
from ultralytics import YOLO
from tracker.DiffMOTtracker import diffmottracker
from diffmot import DiffMOT
from image_processing import image_processing_techniques
from tensorboardX import SummaryWriter
import time
# from torch2trt import TRTModule
from external.YOLOX.yolox.data.data_augment import preproc 
from torchsummary import summary
start_time = time.time()

class ReidTracker:
    def __init__(self):
        pass

    def buildDocker(self):
        config_params = {"eps": 0.001,
            #eval_mode: False
            "eval_mode": True,

            # optimizer
            "lr": 0.0001,
             # train
            "data_dir": '/mnt/8T/home/estar/data/DanceTrack/trackers_gt_GSI',
            "diffnet": "HMINet",
            "interval": 5,
            "augment": False,
            "encoder_dim": 256,
            "tf_layer": 3,
            "epochs": 1200,
            "batch_size": 2048,
            "seed": 123,
            "eval_every": 20,
            "gpus": [0,1,2,3],

            # Testing,
            "eval_at": 1200,
            "det_dir": "/mnt/8T/home/estar/data/DanceTrack/detections/val",
            "info_dir": '/mnt/8T/home/estar/data/DanceTrack/val',
            "reid_dir": "D:/SportsMOT/SportsMOlT/embeddings_yolox_xgfdgfg",
            "save_dir": "/mnt/8T/home/estar/data/DanceTrack/results/val/yolox_m_lt_ddm_1000eps_deeper_800_1rev",
            "eval_expname": "lt_ddm_1000_deeper",
            "high_thres": 0.6,
            "low_thres": 0.4,
            "w_assoc_emb": 2.0,
            "aw_param": 1.2,
            "preprocess_workers": 16,

            # Data Parameters:,
            # device: cuda,
            "device": "cpu",
            "eval_device": None,
            }
        parser = argparse.ArgumentParser(description='DiffMOT Tracker Configuration')
        for param_name, default_value in config_params.items():
            parser.add_argument(f'--{param_name}', type=type(default_value), default=default_value)
        args = parser.parse_args()
        tracker = diffmottracker(  # import diffmottracker
            args,  # Adjust thresholds as needed
            frame_rate=30)
        
        return tracker,args
    
    def get_Detections(self, frame,model_detection):  # object detection using yolov8x
        # ordered_dict  = model_detection['ddpm']
        # model_detection = ordered_dict['module.encoder.cls_token'] 
        # print("Inside get_Detections model_detection : ",model_detection)
        results = model_detection(frame,
                               save=False,
                               conf=0.5,
                               iou=0.8,
                               classes=0,
                               imgsz=1280)
        # cv2.imshow("l",results[0].plot())
        # cv2.waitKey(200)
        # cv2.destroyAllWindows()
        return results
    
    # Function to remove 'module.' prefix from state_dict keys
    def remove_module_prefix(self,state_dict):
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v  # Remove 'module.' prefix
            else:
                new_state_dict[k] = v
        return new_state_dict



ReidTracker_obj=ReidTracker() # initiate tracker
tracker, config_ = ReidTracker_obj.buildDocker()
sports_model_8 = YOLO('yolov8x.pt')
# Detection model

# sports_model= torch.load("D:/Diffusion_based_MOT/_epoch1200.pt",map_location = "cpu" )

# sports_model= torch.load("C:/Users/Acer/Downloads/SportsMOT_epoch1200.pt",map_location = "cpu" )
sports_model= torch.load("C:/Users/Acer/Downloads/MOT_epoch800.pt",map_location = "cpu" )
# Remove 'module.' prefix
# sports_model = ReidTracker_obj.remove_module_prefix(sports_model)

# print("sports_model : ",sports_model)
# sports_model = sports_model["ddpm"]["OrderedDict"]
print(sports_model.keys())
# sports_model = sports_model__["ddpm"]
diffmot_obj = DiffMOT(config_)
model = diffmot_obj._build_model()
# model = yolox_custom()
model.load_state_dict(sports_model,strict=False)
model = model.eval()

# frame = cv2.imread("D:/Diffusion_based_MOT/custom_model_output/det_root/1.png")
# preproc_frame,_ = preproc(frame,(720,1280))
# frame = torch.from_numpy(preproc_frame).unsqueeze(0)
# frame = frame.float()
# a = model(frame)
# print("a : ",a)
# a


# print("------------------- :",SummaryWriter(model))
# print("sports_model : ",summary(model, input_size=(3,720, 1280)))
# missing_keys, unexpected_keys = model.load_state_dict(sports_model,strict=False)

v_path="D:/Diffusion_based_MOT/4_Line_Warm_Up.mp4"
cap = cv2.VideoCapture(v_path)
total_frames = (cap.get(cv2.CAP_PROP_FRAME_COUNT))
print("total_frames : ",total_frames)
make_video = cv2.VideoWriter("badmenton.mp4", cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 20, (int(cap.get(3)), int(cap.get(4))))
img_processing_tech=image_processing_techniques() # use to get video info and make video

c=0
while True:
    ret, frame = cap.read()
    c=c+1
    if(c<=520):
        continue
    if(c>740):
        break
    # if(c%10==0):
    print("--------------- c ------------------ : ",c)
    if not ret:
      break

    # preproc_frame,_ = preproc(frame,(720,1280))
    # frame = torch.from_numpy(preproc_frame)#.unsqueeze(0)
    # frame = frame.float()

    results = ReidTracker_obj.get_Detections(frame,sports_model_8)       # object detection
    results
    # detections=results[0].boxes.data                                            
    h,w=frame.shape[:2]
    output = tracker.update(results,model, c,w,h,"tag",frame)       # update the tracker
    img_processing_tech.make_video_from_tracker(frame,output,make_video)


end_time = time.time()
inference_time = end_time - start_time
print(f"Inference time: {inference_time} seconds")



