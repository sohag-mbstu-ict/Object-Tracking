import cv2
import numpy as np
import math
from shapely.geometry import Polygon, Point

class select_lost_or_removed_id:
    def __init__(self):
        pass
    
    def calculate_distance_from_bbox_center_to_court_end(self,court_end,bbox_center):
        distance = math.sqrt((court_end[0] - bbox_center[0]) ** 2 + (court_end[1] - bbox_center[1]) ** 2)
        return distance
        
        
    def select_first_or_second_closest_lost_stracks(self,track):
        second_closest_court_left_end=(442,153)
        second_closest_court_right_end=(818,153)
        first_closest_court_left_end=(280,520)
        first_closest_court_right_end=(1016,520)
        
        bbox=track.tlbr
        x1,y1,x2,y2=bbox[0],bbox[1],bbox[2],bbox[3]
        bbox_center=((x1+x2)/2, y2)

        second_closest_court_left_end_dist=self.calculate_distance_from_bbox_center_to_court_end(second_closest_court_left_end,bbox_center)
        second_closest_court_right_end_dist=self.calculate_distance_from_bbox_center_to_court_end(second_closest_court_right_end,bbox_center)
        second_closest_court_min=min(second_closest_court_left_end_dist,second_closest_court_right_end_dist)
        
        first_closest_court_left_end_dist=self.calculate_distance_from_bbox_center_to_court_end (first_closest_court_left_end,bbox_center)
        first_closest_court_right_end_dist=self.calculate_distance_from_bbox_center_to_court_end(first_closest_court_right_end,bbox_center)
        first_closest_court_min=min(first_closest_court_left_end_dist,first_closest_court_right_end_dist)
        print("first_closest_court_min  : ",first_closest_court_min)
        print("second_closest_court_min : ",second_closest_court_min)
        if(first_closest_court_min>second_closest_court_min):
            print(" Return First Half Return First Half Return First Half")
            return "second_half"
        else:
            print(" Second  Half Return Second Half Return Second Half")
            return "first_half"
    
    
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

        if(abs(prev_avg_width-pres_avg_width)>=70):
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
    

   