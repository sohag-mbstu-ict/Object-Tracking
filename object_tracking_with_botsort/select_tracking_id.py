import cv2
import numpy as np

class select_lost_or_removed_id:
    def __init__(self):
        pass

    
    def calculate_iou(self,box1,bbox_all):
        x1_1, y1_1, x2_1, y2_1 = box1[0],box1[1],box1[2],box1[3]
        iou=-8
        for j in range(0,len(bbox_all)):
            box2=bbox_all[j]
            # if(np.any(box1==0) or np.any(box2==0)):
            #     continue
            # Extract coordinates
            x1_2, y1_2, x2_2, y2_2 = box2[0],box2[1],box2[2],box2[3]
            # Calculate the coordinates of the intersection area
            x_intersection = max(0, min(x2_1, x2_2) - max(x1_1, x1_2))
            y_intersection = max(0, min(y2_1, y2_2) - max(y1_1, y1_2))
            area_intersection = x_intersection * y_intersection
            # Calculate the area of each bounding box
            area_box1 = (x2_1 - x1_1) * (y2_1 - y1_1)
            area_box2 = (x2_2 - x1_2) * (y2_2 - y1_2)
            # Calculate the area of the union
            area_union = area_box1 + area_box2 - area_intersection
            # Calculate IoU
            iou = area_intersection / max(1e-10, area_union)
            return iou
        
    def calculate_lost_stracks_removed_stracks_iou(self,box1,bbox_all):
        x1_1, y1_1, x2_1, y2_1 = box1[0],box1[1],box1[2],box1[3]
        max_iou=0
        index=-3
        for j in range(0,len(bbox_all)):
            box2=bbox_all[j]
            # if(np.any(box1==0) or np.any(box2==0)):
            #     continue
            # Extract coordinates
            x1_2, y1_2, x2_2, y2_2 = box2[0],box2[1],box2[2],box2[3]
            # Calculate the coordinates of the intersection area
            x_intersection = max(0, min(x2_1, x2_2) - max(x1_1, x1_2))
            y_intersection = max(0, min(y2_1, y2_2) - max(y1_1, y1_2))
            area_intersection = x_intersection * y_intersection
            # Calculate the area of each bounding box
            area_box1 = (x2_1 - x1_1) * (y2_1 - y1_1)
            area_box2 = (x2_2 - x1_2) * (y2_2 - y1_2)
            # Calculate the area of the union
            area_union = area_box1 + area_box2 - area_intersection
            # Calculate IoU
            iou = area_intersection / max(1e-10, area_union)
            if(max_iou<iou):
                max_iou=iou
                index=j
            return max_iou, index


    def select_correct_tracking_id(self,track,bbox,lost_stracks,removed_stracks):
        lost_flag_=False
        removed_iou=0
        lost_iou=0
        removed_flag_=False
        if(len(lost_stracks)>0): #calculate iou with lost_stracks
            lost_list=[]
            for i in lost_stracks:
                lost_list.append(i.tlbr)

            lost_iou, lost_index = self.calculate_lost_stracks_removed_stracks_iou(bbox,lost_list)
            if(lost_iou>0):
                lost_flag_=True

        if(len(removed_stracks)>0): #calculate iou with removed_stracks
            removed_list=[]
            for i in removed_stracks:
                removed_list.append(i.tlbr)

            removed_iou, removed_index = self.calculate_lost_stracks_removed_stracks_iou(bbox,removed_list)
            if(removed_iou>0):
                removed_flag_=True

        if(lost_flag_==True or removed_flag_==True):
            if(removed_iou>lost_iou):
                track.track_id=removed_stracks[removed_index].track_id
            else:
                track.track_id=lost_stracks[lost_index].track_id

        else:
            if(len(lost_stracks)>0):
                a=len(lost_stracks)
                track.track_id=lost_stracks[a-1].track_id
            else:
                a=len(removed_stracks)
                if(a>0):
                    track.track_id=removed_stracks[a-1].track_id

        return track


    def handle_occlusion(self,result):
        bbox=result.boxes.xyxy.tolist()
        scores=result.boxes.conf
        cls=result.boxes.cls
        
        for i in range(len(bbox)-1):
            box1=bbox[i]
            for j in range(i+1,len(bbox)):
                box2=bbox[j]
                if(np.any(box1==0) or np.any(box2==0)):
                    continue
                # Extract coordinates
                x1_1, y1_1, x2_1, y2_1 = box1[0],box1[1],box1[2],box1[3]
                x1_2, y1_2, x2_2, y2_2 = box2[0],box2[1],box2[2],box2[3]
                # Calculate the coordinates of the intersection area
                x_intersection = max(0, min(x2_1, x2_2) - max(x1_1, x1_2))
                y_intersection = max(0, min(y2_1, y2_2) - max(y1_1, y1_2))
                area_intersection = x_intersection * y_intersection
                # Calculate the area of each bounding box
                area_box1 = (x2_1 - x1_1) * (y2_1 - y1_1)
                area_box2 = (x2_2 - x1_2) * (y2_2 - y1_2)
                # Calculate the area of the union
                area_union = area_box1 + area_box2 - area_intersection
                # Calculate IoU
                iou = area_intersection / max(1e-10, area_union)
                print("iou : ",iou)
                if(iou>0.2):
                    if(scores[i]>scores[j]):
                        bbox[j]=0
                    else:
                        bbox[i]=0
        dets_=[]
        scores_=[]
        cls_=[]
        for i in range(len(bbox)):
            if np.any(bbox[i] != 0):
                dets_.append(bbox[i])
                scores_.append(scores[i])
                cls_.append(cls[i])

        return dets_,scores_,cls_
            
            



