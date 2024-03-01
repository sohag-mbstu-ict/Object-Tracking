import pandas as pd
from ultralytics import YOLO
from ultralytics.solutions import object_counter
from image_processing import image_processing_techniques
from jersey_color.find_jersey_color_another import find_jersey_color
from bbox_id_store_in_json import store_bbox_id_number_in_json_csv_file
from football_and_player_activities.football_bbox_extraction import football_detection
from ultralytics.trackers.bot_sort import BOTrack,BYTETracker
import argparse

# Configuration parameters
config_params = {
    'tracker_type': 'botsort',
    'track_high_thresh': 0.3,
    'track_low_thresh': 0.1,
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

# Access the arguments using dot notation (e.g., args.tracker_type)

import cv2
import time
start_time = time.time()


model = YOLO('yolov8x.pt')
cap = cv2.VideoCapture("C:/Users/Acer/Downloads/0221.mp4")
# cap = cv2.VideoCapture("E:/dataset/long_pass/trimed_3.mkv")

total_frames = (cap.get(cv2.CAP_PROP_FRAME_COUNT))
print("total_frames : ",total_frames)
assert cap.isOpened(), "Error reading video file"

# Define region points
region_points = [(0, 400), (1800, 404), (1800, 360), (0, 360)]
classes_to_count = [0]
img_processing_tech=image_processing_techniques()
cap, frame_width, frame_height = img_processing_tech.get_video_info("C:/Users/Acer/Downloads/0221.mp4")
# cap, frame_width, frame_height = img_processing_tech.get_video_info("E:/dataset/long_pass/trimed_3.mkv")
# Video writer
output = cv2.VideoWriter("E:/dataset/long_pass/022_sift.mp4", cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))

# # Init Object Counter
counter = object_counter.ObjectCounter()
counter.set_args(view_img=True,
                 reg_pts=region_points,
                 classes_names=model.names,
                 draw_tracks=True)

# Create a background subtractor object
background_subtractor = cv2.createBackgroundSubtractorMOG2()
df = pd.DataFrame(columns=['bbox','tracking_id'])
frame_count=0
c=0

find_jersey_color_obj=find_jersey_color() #jersey color extraction
obj_csv_json=store_bbox_id_number_in_json_csv_file() #class instance to store info in csv file
# obj_json.store_bbox_tracking_id_number_in_csv("initialize") # store in csv file

football_bbox_obj=football_detection() # to extract football bbox
df_football_bbox = pd.DataFrame(columns=['bbox','bbox_center'])
frame_count=0

with open('tracking_id.txt', 'w') as file:
    file.write('')
with open('tracking_id_tlbr.txt', 'w') as file:
    file.write('')

tracker = BYTETracker(
     args,  # Adjust thresholds as needed
     frame_rate=30)


while cap.isOpened():
    c+=1
    if(c>650):
        break
    if(c%50==0):
        print("frame number : ",c)
    success, im0 = cap.read()
    if(c<=80):
        continue

    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break
    # fg_result = img_processing_tech.background_Substitution(im0,background_subtractor)
    fg_result = img_processing_tech.color_Enhancement(im0)
    # det_result = model.predict(fg_result,save=False,conf=0.5,imgsz=1280)


    # for r in det_result[0]:
    #     det_bboxes=r.boxes.xyxy.tolist()
    #     x1, y1, x2, y2 = det_bboxes[0]
    #     bbox_cls=r.boxes.conf.tolist()
    #     cropped_img = fg_result[int(y1):int(y2), int(x1):int(x2)]
    #     detections = [[bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1], 1.0] for bbox in det_bboxes]

    #     bbox_cls = [[bbox[0], 1.0] if isinstance(bbox, (list, tuple)) else [bbox, 1.0] for bbox in bbox_cls]

    #     tracker.update(det_result[0], detections,bbox_cls,det_cls=[],img=fg_result)
        

    # print("det_bboxes : ",det_result)


    tracks_output = model.track(fg_result, 
                        persist=True,
                        verbose=False,
                        conf=0.3, 
                        iou=0.7,
                        show=True,                         
                        classes=classes_to_count,
                        tracker="botsort.yaml"       
                        )

        # cv2.imshow("c",tracks_output[0].plot())
        # cv2.waitKey(200)
        # cv2.destroyAllWindows()

    # output.write(tracks_output[0].plot())

# df_football_bbox.to_csv('Purtogul.csv', encoding='utf-8')

# df.to_csv('tracking_id.csv', encoding='utf-8')
end_time = time.time()
inference_time = end_time - start_time
print(f"Inference time: {inference_time} seconds")

cap.release()
output.release()
cv2.destroyAllWindows()

