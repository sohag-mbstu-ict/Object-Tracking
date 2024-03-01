import pandas as pd
from ultralytics import YOLO
from image_processing import image_processing_techniques
from jersey_color.find_jersey_color_another import find_jersey_color
from bbox_id_store_in_json import store_bbox_id_number_in_json_csv_file
from football_and_player_activities.football_bbox_extraction import football_detection
from select_tracking_id import select_lost_or_removed_id
from ultralytics.trackers.bot_sort import BOTrack,BYTETracker
import argparse

# Configuration parameters
config_params = {
    'tracker_type': 'botsort',
    'track_high_thresh': 0.6,
    'track_low_thresh': 0.35,
    'new_track_thresh': 0.75,
    'track_buffer': 300,
    'match_thresh': 0.9,
    # Add more parameters as needed
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
v_path="C:/Users/Acer/Downloads/0221.mp4"
cap = cv2.VideoCapture(v_path)
# cap = cv2.VideoCapture("E:/dataset/long_pass/trimed_3.mkv")

total_frames = (cap.get(cv2.CAP_PROP_FRAME_COUNT))
print("total_frames : ",total_frames)
assert cap.isOpened(), "Error reading video file"

# Define region points
region_points = [(0, 400), (1800, 404), (1800, 360), (0, 360)]
classes_to_count = [0]
img_processing_tech=image_processing_techniques()
cap, frame_width, frame_height = img_processing_tech.get_video_info(v_path)
# cap, frame_width, frame_height = img_processing_tech.get_video_info("E:/dataset/long_pass/trimed_3.mkv")
# Video writer
make_video = cv2.VideoWriter("E:/dataset/long_pass/022_sift.mp4", cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))


# Create a background subtractor object
background_subtractor = cv2.createBackgroundSubtractorMOG2()
cal_iou_obj = select_lost_or_removed_id()


tracker = BYTETracker(
     args,  # Adjust thresholds as needed
     frame_rate=30)

c=0
while cap.isOpened():
    c+=1
    if(c>620):
        break
    if(c%50==0):
        print("frame number : ",c)
    success, im0 = cap.read()
    if(c<=70):
        continue

    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break
    # fg_result = img_processing_tech.background_Substitution(im0,background_subtractor)
    fg_result = img_processing_tech.color_Enhancement(im0)


    det_result = model.predict(fg_result,
                               save=False,
                               conf=0.3,
                               iou=0.85,
                               classes=0,
                               imgsz=1280)
    
    dets_,scores_,cls_ = cal_iou_obj.handle_occlusion(det_result[0])

    # print("Inside botsort_main.py dets_,scores_,cls_  : ",dets_,scores_,cls_ )

    output=tracker.update(det_result[0], dets_, scores_,cls_,img=fg_result)  
    output

    img_processing_tech.make_video_from_tracker(fg_result,output,make_video)
        

# df.to_csv('tracking_id.csv', encoding='utf-8')
end_time = time.time()
inference_time = end_time - start_time
print(f"Inference time: {inference_time} seconds")

# output.release()
cv2.destroyAllWindows()

