import pandas as pd
import numpy as np
import torch
from ultralytics import YOLO
from image_processing import image_processing_techniques
from find_jersey_color import find_jersey_color
from sklearn.cluster import KMeans
from bbox_id_store_in_json import store_bbox_id_number_in_json_csv_file
from football_and_player_activities.football_bbox_extraction import football_detection
import webcolors
from ultralytics.trackers.bot_sort import BOTSORT,BYTETracker,BOTrack

import torchvision.transforms as transforms
import argparse
# from tensorflow.keras.applications import EfficientNetB0
# reid_model = efficientnet_b0(pretrained=True)

efficientnet = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b0', pretrained=True)
utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_convnets_processing_utils')



# Configuration parameters
config_params = {
    'tracker_type': 'botsort',
    'track_high_thresh': 0.5,
    'track_low_thresh': 0.35,
    'new_track_thresh': 0.75,
    'track_buffer': 300,
    'match_thresh': 0.9,
    # BoT-SORT settings
    'gmc_method': 'sift', 
    # ReID model related thresh (not supported yet)
    'proximity_thresh': 0.7, 
    'appearance_thresh': 0.8,
    'with_reid': True
}

# Create an argument parser
parser = argparse.ArgumentParser(description='YOLO Tracker Configuration')

# Add command-line arguments based on the configuration parameters
for param_name, default_value in config_params.items():
    parser.add_argument(f'--{param_name}', type=type(default_value), default=default_value)

# Parse the command-line arguments
args = parser.parse_args()

import cv2
import time
start_time = time.time()


model = YOLO('yolov8x.pt')
v_path="E:/dataset/football_video/france.mkv"
# v_path="E:/dataset/football_video/trimed_pass.mkv"
cap = cv2.VideoCapture(v_path)

total_frames = (cap.get(cv2.CAP_PROP_FRAME_COUNT))
print("total_frames : ",total_frames)
assert cap.isOpened(), "Error reading video file"

# Define region points
region_points = [(0, 400), (1800, 404), (1800, 360), (0, 360)]
classes_to_count = [0]
img_processing_tech=image_processing_techniques()
cap, frame_width, frame_height = img_processing_tech.get_video_info(v_path)

cap = cv2.VideoCapture(v_path)
# Video writer
make_video = cv2.VideoWriter("E:/dataset/long_pass/football_f.mp4", cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (1280, 736))
def preprocess_frame(frame):
  # Resize, normalize, and convert to tensor format
  resized_frame = cv2.resize(frame, (1280, 736))
  normalized_frame = resized_frame / 255.0
  tensor_frame = torch.from_numpy(normalized_frame.transpose((2, 0, 1))).unsqueeze(0)
  return tensor_frame


def extract_features(detections,tensor_frame):
  # Crop bounding boxes and extract features with reid_model
  features = []
  # print("tensor_frame Shape:", tensor_frame.shape)
  for box in detections:
    xmin, ymin, xmax, ymax = int(box[0]),int(box[1]),int(box[2]),int(box[3])
    cropped_frame = tensor_frame[:,:, ymin:ymax, xmin:xmax]

    # print("Cropped Frame Shape:", cropped_frame.shape)
    cropped_frame=cropped_frame.float()
    feature = efficientnet(cropped_frame)
    features.append(feature)
  return torch.cat(features, dim=0)


tracker = BYTETracker(    # import BYTETracker
            args,         # Adjust thresholds as needed
            frame_rate=30)

# Create a background subtractor object
background_subtractor = cv2.createBackgroundSubtractorMOG2()
jersey_color_obj = find_jersey_color()
c=0
while True:
  ret, frame = cap.read()
  c=c+1
  if not ret:
    break
  if(c<=80):
    continue
  if(c>=350):
    break
  if(c%10==0):
    print("--------------- c ------------------ : ",c)
  frame = cv2.resize(frame, (1280, 736))

  fg_result = img_processing_tech.color_Enhancement(frame) # apply image processing technique
  results_color = model.predict  (    fg_result, # apply yolov8x for jersey color 
                                      save=False,
                                      conf=0.5,
                                      iou=0.8,
                                      classes=0,
                                      imgsz=1280)
  
  color_bbox, jersey_color_img=jersey_color_obj.find_color(frame, results_color[0].boxes.xyxy, threshold=0)
  fixed_color_jersey=[]

  det_xyxy=[]
  det_conf=[]
  det_cls=[]
  for index_, color_ in enumerate(color_bbox):
    if(color_=='red' or color_=='blue' or color_=='black' or color_=='white'):
      fixed_color_jersey.append(results_color[0].boxes.data[index_])
      det_xyxy.append(results_color[0].boxes.xyxy[index_])
      det_conf.append(results_color[0].boxes.conf[index_])
      det_cls.append(results_color[0].boxes.cls[index_])

  fixed_color_jersey=np.array(fixed_color_jersey) # only assigned color jersey will pass


  if(len(fixed_color_jersey)>0):
    # Preprocess frame
    tensor_frame = preprocess_frame(fg_result)
    detections = fixed_color_jersey
    # # Feature extraction with EfficientNet-B0
    features = extract_features(detections,tensor_frame)
    # print("len(conf) : ",len(det_conf))
    # Update tracker with detections and features
    output = tracker.update(results_color[0],det_xyxy,det_conf,det_cls, features.detach().cpu().numpy())
    img_processing_tech.make_video_from_tracker(frame,output,make_video,color_bbox) # make output video




end_time = time.time()
inference_time = end_time - start_time
print(f"Inference time: {inference_time} seconds")




  # bboxes = results[0].boxes.xyxy
  # # bboxes = np.concatenate([bboxes, np.arange(len(bboxes)).reshape(-1, 1)], axis=-1)
  # scores = results[0].boxes.conf
  # cls = results[0].boxes.cls
  # bot_sort_tracker = BOTSORT(args, frame_rate=30)
  # bo_track = BOTrack(bboxes, scores, cls, features.detach().cpu().numpy())
  # bo_track.predict()
  # bo_track.update(new_track, frame_id)
  # tracks=bot_sort_tracker.init_track(bboxes, scores, cls, features.detach().cpu().numpy())
  # output=bot_sort_tracker.multi_predict(tracks)
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  