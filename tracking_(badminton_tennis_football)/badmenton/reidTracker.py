import cv2
import numpy as np
from image_processing import image_processing_techniques
from tinydb import TinyDB
from ultralytics import YOLO
import time
import timm
import torch
from ultralytics.trackers.bot_sort import BYTETracker
import argparse
from shapely.geometry import Polygon
start_time = time.time()

class ReidTracker:
    def __init__(self,tracker,efficientnet):
        self.tracker = self.buildDocker()
        self.tracked_objects = {}
        self.efficientnet = efficientnet

    def buildDocker(self):
        config_params = {
            'tracker_type': 'botsort',
            'track_high_thresh': 0.7,
            'track_low_thresh': 0.7,
            'new_track_thresh': 0.75,
            'track_buffer': 520,
            'match_thresh': 0.9,
            'gmc_method': 'sift',
            # ReID model related thresh (not supported yet)
            'proximity_thresh': 0.6,
            'appearance_thresh': 0.35,
            'with_reid': True
        }
        parser = argparse.ArgumentParser(description='YOLO Tracker Configuration')
        for param_name, default_value in config_params.items():
            parser.add_argument(f'--{param_name}', type=type(default_value), default=default_value)
        args = parser.parse_args()
        tracker = BYTETracker(  # import BYTETracker
            args,  # Adjust thresholds as needed
            frame_rate=30)
        return tracker
        
    def extract_features(self,detections, tensor_frame):
        features = []
        for box in detections:
            xmin, ymin, xmax, ymax = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            cropped_frame = tensor_frame[:, :, ymin:ymax, xmin:xmax]
            cropped_frame = cropped_frame.float()
            feature = self.efficientnet(cropped_frame)
            features.append(feature)
        return torch.cat(features, dim=0)
    
    def preprocess_frame(self,frame):
        # Resize, normalize, and convert to tensor format
        resized_frame = cv2.resize(frame, (1280, 736))
        normalized_frame = resized_frame / 255.0
        tensor_frame = torch.from_numpy(normalized_frame.transpose((2, 0, 1))).unsqueeze(0)
        return tensor_frame

    def update_tracker(self, result, detections_bbox,det_conf,det_cls,features):
        return self.tracker.update(result,detections_bbox,det_conf,det_cls,features)
    
    def get_Detections(self, tensor_frame,model_detection):  # object detection using yolov8x
        results = model_detection.predict(tensor_frame,   
                               save=False,
                               conf=0.5,
                               iou=0.8,
                               classes=0,
                               imgsz=1280)
        return results
    
    def get_Keypoints(self, cropped_img,model_pose):         # model_pose = yolov8n-pose.pt
        result = model_pose.predict(cropped_img, 
                                    save=False,
                                    conf=0.1)
        keypoints=result[0].keypoints.xy.tolist()
        return keypoints
    
    def getTrackerID_bbox_keypoints(self, output,img,model_pose,db_table): 
        list_=[]
        for box in output.T.base:
            dict_={}
            xmin, ymin,xmax,ymax=int(box[0]), int(box[1]), int(box[2]), int(box[3])
            cropped_img = img[ymin:ymax, xmin:xmax]
            keypoints=self.get_Keypoints(cropped_img,model_pose) # extract keypoints
            track_id=int(box[4])
            dict_["Tracking_ID"]=track_id
            dict_["bbox"]=list((xmin, ymin,xmax,ymax))
            dict_["keypoints"]=keypoints
            list_.append(dict_)      
        db_table.insert_multiple(list_)
        return db_table
        
    def remove_false_positive_detections_using_court_coordinates(self,results):
        bboxes=results[0].boxes.xyxy.detach().cpu().numpy()
        conf=results[0].boxes.conf.detach().cpu().numpy()
        cls=results[0].boxes.cls.detach().cpu().numpy()
        court = Polygon([(445,188), (825,188), (1120,560), (160, 560)])
        bbox_=[]
        conf_=[]
        cls_=[]
        for i,bbox in enumerate(bboxes):
            x1,y1,x2,y2=bbox[0],bbox[1],bbox[2],bbox[3]
            bbox_polygon=Polygon([(x1,y1),(x2,y1),(x2,y2),(x1,y1)])
            do_overlap = court.intersects(bbox_polygon) # Check if polygons overlap
            if do_overlap:
                bbox_.append(bboxes[i])
                conf_.append(conf[i])
                cls_.append(cls[i])
        return bbox_,conf_,cls_

model_detection = YOLO('yolov8x.pt')

v_path="/home/azureuser/data/datadisk/re_ID_Tracker/input_video/4_tennis_player/4_5.mp4"

cap = cv2.VideoCapture(v_path)
total_frames = (cap.get(cv2.CAP_PROP_FRAME_COUNT))
print("total_frames : ",total_frames)
make_video = cv2.VideoWriter("/home/azureuser/data/datadisk/re_ID_Tracker/out_video/tennis_4_player/ten_4_5.mp4", cv2.VideoWriter_fourcc(*'MP4V'), 20, (int(cap.get(3)), int(cap.get(4))))
img_processing_tech=image_processing_techniques() # use to get video info and make video



# model_path = "C:/Users/Acer/Downloads/files/nvidia_efficientnet-b0_210412.pth"
model_path = "/home/azureuser/data/datadisk/re_ID_Tracker/efficientnet_b0/tf_efficientnet_b1_ns-99dd0c41.pth"
model_name = "efficientnet_b1"
efficientnet= timm.create_model(model_name, pretrained=False,num_classes=1000)
state_dict = torch.load(model_path, map_location=torch.device('cpu'))  # Use 'cuda' if GPU available
efficientnet.load_state_dict(state_dict, strict=False)

ReidTracker_obj=ReidTracker('bytetracker',efficientnet) # initiate tracker

db = TinyDB('pose.json')
db_table = db.table('db_table')
db_table.truncate()
frame_count=0


while True:
  ret, frame = cap.read()
  frame_count=frame_count+1
#   if(frame_count<=600):
#       continue
  if(frame_count>13333):
      break
  if(frame_count%10==0):
    print("--------------- c ------------------ : ",frame_count)
  if not ret:
    break

  tensor_frame = ReidTracker_obj.preprocess_frame(frame)                       # Preprocess frame
  results = ReidTracker_obj.get_Detections(tensor_frame,model_detection)       # object detection
  
  detections_bbox,det_conf,det_cls=ReidTracker_obj.remove_false_positive_detections_using_court_coordinates(results)  # remove false positive detections                                      
  
  if(len(det_conf)>0): 
    features = ReidTracker_obj.extract_features(detections_bbox,tensor_frame)         # extract feature using efficientnet
    print("len(features : ",len(features),"   : ",len(results[0].boxes.conf))
    if(len(features)>0):
        output = ReidTracker_obj.update_tracker(results[0],detections_bbox,det_conf,det_cls, features.detach().cpu().numpy())       # update the tracker
        # db_table=ReidTracker_obj.getTrackerID_bbox_keypoints(output,frame,model_pose,db_table)   # store info in json file
        img_processing_tech.make_video_from_tracker(frame,output,make_video)                       # make output video

make_video.release()
end_time = time.time()
inference_time = end_time - start_time
print(f"Inference time: {inference_time} seconds")
