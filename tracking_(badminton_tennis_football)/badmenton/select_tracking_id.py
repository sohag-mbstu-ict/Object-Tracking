import cv2
import numpy as np
from shapely.geometry import Polygon, Point

class select_lost_or_removed_id:
    def __init__(self):
        pass

        
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

    



