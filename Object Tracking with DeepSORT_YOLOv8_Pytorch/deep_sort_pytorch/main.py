import cv2
import torch
import numpy as np
import math
from numpy import random
# from IPython.display import HTML
from base64 import b64encode
import os
from ultralytics import YOLO
from utils.parser import get_config
from deep_sort import DeepSort


def get_video_info(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if the video file is opened successfully
    if not cap.isOpened():
        raise ValueError("Error: Could not open the video file.")

    # Get the frame width and frame height
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    return cap, frame_width, frame_height

video_path = 'E:/dataset/run/5_.mp4'
cap, frame_width, frame_height = get_video_info(video_path)
print("cap, frame_width, frame_height : ", frame_width, frame_height)


def cococlassNames():
  class_names = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella","handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat","baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup","fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli","carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed","diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone","microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors","teddy bear", "hair drier", "toothbrush" ]
  return class_names

classNames = cococlassNames()




def initialize_deepsort():
    # Create the Deep SORT configuration object and load settings from the YAML file
    cfg_deep = get_config()
    cfg_deep.merge_from_file("E:/deepSORT_yolov8_Moin/YOLOv8-DeepSORT-Object-Tracking/deep_sort_pytorch/configs/deep_sort.yaml")

    # Initialize the DeepSort tracker
    deepsort = DeepSort(cfg_deep.DEEPSORT.REID_CKPT,
                        max_dist=cfg_deep.DEEPSORT.MAX_DIST,
                        # min_confidence  parameter sets the minimum tracking confidence required for an object detection to be considered in the tracking process
                        min_confidence=cfg_deep.DEEPSORT.MIN_CONFIDENCE,
                        #nms_max_overlap specifies the maximum allowed overlap between bounding boxes during non-maximum suppression (NMS)
                        nms_max_overlap=cfg_deep.DEEPSORT.NMS_MAX_OVERLAP,
                        #max_iou_distance parameter defines the maximum intersection-over-union (IoU) distance between object detections
                        max_iou_distance=cfg_deep.DEEPSORT.MAX_IOU_DISTANCE,
                        # Max_age: If an object's tracking ID is lost (i.e., the object is no longer detected), this parameter determines how many frames the tracker should wait before assigning a new id
                        max_age=cfg_deep.DEEPSORT.MAX_AGE, n_init=cfg_deep.DEEPSORT.N_INIT,
                        #nn_budget: It sets the budget for the nearest-neighbor search.
                        nn_budget=cfg_deep.DEEPSORT.NN_BUDGET,
                        use_cuda=True
        )

    return deepsort


def cococlassNames():
  class_names = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella","handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat","baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup","fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli","carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed","diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone","microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors","teddy bear", "hair drier", "toothbrush" ]
  return class_names

names = cococlassNames()
colors = [[random.randint(0, 255) for _ in range(3)]
          for _ in range(len(names))]
palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)

def compute_color_for_labels(label):
    """
    Function that adds fixed color depending on the class
    """
    if label == 0:  # person  #BGR
        color = (85, 45, 255)
    elif label == 2:  # Car
        color = (222, 82, 175)
    elif label == 3:  # Motobike
        color = (0, 204, 255)
    elif label == 5:  # Bus
        color = (0, 149, 255)
    else:
        color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


#Creating a helper function

def draw_boxes(img, bbox, identities=None, categories=None, names=None, offset=(0,0)):
    for i, box in enumerate(bbox):
        x1, y1,  x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[0]
        y2 += offset[0]
        cat = int(categories[i]) if categories is not None else 0
        id = int(identities[i]) if identities is not None else 0
        #Create Bounding Boxes around the Detected Objects
        cv2.rectangle(img, (x1, y1), (x2, y2), color= compute_color_for_labels(cat),thickness=2, lineType=cv2.LINE_AA)
        label = str(id) + ":" + classNames[cat]
        # print("label : ",label)
        (w,h), _ = cv2.getTextSize(str(label), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1/2, thickness=1)
        #Create a rectangle above the detected object and add label and confidence score
        t_size=cv2.getTextSize(str(label), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1/2, thickness=1)[0]
        c2=x1+t_size[0], y1-t_size[1]-3
        cv2.rectangle(img, (x1, y1), c2, color=compute_color_for_labels(cat), thickness=-1, lineType=cv2.LINE_AA)
        cv2.putText(img, str(label), (x1, y1-2), 0, 1/2, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)
    return img





deepsort = initialize_deepsort()
model = YOLO('yolov8n.pt')
model.predict("C:/Users/Acer/Downloads/image.png")

video_path = 'E:/dataset/run/5_.mp4'
cap, frame_width, frame_height = get_video_info(video_path)

classNames = cococlassNames()
output = cv2.VideoWriter('E:/dataset/run/Output12.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))
c=0
import time
start_time = time.time()
cap = cv2.VideoCapture(video_path)
model = YOLO("yolov8n.pt")
while True:
    c+=1
    xywh_bboxs = []
    confs = []
    oids = []
    outputs = []
    ret, frame = cap.read()
    if ret:
      result = model.predict(frame, conf=0.5)
      bbox_xyxys=result[0].boxes.xyxy.tolist()
      confidences=result[0].boxes.conf.tolist()
      labels=result[0].boxes.cls.tolist()
      for (bbox_xyxy, confidence, cls) in zip(bbox_xyxys, confidences, labels):
          bbox = np.array(bbox_xyxy)
          x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
          x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
          conf = math.ceil((confidence*100))/100
          cx, cy = int((x1+x2)/2), int((y1+y2)/2)
          bbox_width = abs(x1-x2)
          bbox_height = abs(y1-y2)
          xcycwh = [cx, cy, bbox_width, bbox_height]
          xywh_bboxs.append(xcycwh)
          confs.append(conf)
          oids.append(int(cls))
      xywhs = torch.tensor(xywh_bboxs)
      confss= torch.tensor(confs)
      if(len(xywhs)>0):
        outputs = deepsort.update(xywhs, confss, oids, frame)
      if len(outputs)>0:
          bbox_xyxy = outputs[:,:4]
          identities = outputs[:, -2]
          object_id = outputs[:, -1]
          frame = draw_boxes(frame, bbox_xyxy, identities, object_id)
      output.write(frame)
    else:
        break

end_time = time.time()
inference_time = end_time - start_time
print(f"Inference time: {inference_time} seconds")

output.release()
cap.release()




