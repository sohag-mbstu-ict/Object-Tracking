import cv2
import numpy as np
import math
from shapely.geometry import Polygon, Point

class select_lost_or_removed_id:
    def __init__(self):
        pass


    
    def handle_scene_change(self,previous_det_list,present_det_list):
        prev_total_width=0
        cnt=0
        for bbox in previous_det_list:
            prev_total_width+= (bbox[2]-bbox[0])
            cnt+=1
        prev_avg_width=prev_total_width/cnt
        
        pres_total_width=0
        cnt=0
        for bbox in present_det_list:
            pres_total_width+= (bbox[2]-bbox[0])
            cnt+=1
        pres_avg_width=pres_total_width/cnt
        
        # print("inside select tracking prev_avg_width  : ",prev_avg_width)
        # print("inside select tracking prev_avg_width  : ",pres_avg_width)
        print("inside select tracking difference      : ",abs(prev_avg_width-pres_avg_width))
        if(abs(prev_avg_width-pres_avg_width)>=110):
          return True
        else:
          return False
        
        
        
    def add_lost_track_in_custom_lost_track(self,track):
        court = Polygon([(327,255), (963,236), (1132, 639), (130, 613)])
        bbox=track.tlbr
        x1, y1, x2, y2 = bbox[0],bbox[1],bbox[2],bbox[3]
        bl=Point(x1,y2)
        br=Point(x2,y2)
        if(court.contains(bl) or court.contains(br)):
            return True
        else:
            return False