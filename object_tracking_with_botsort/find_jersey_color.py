import cv2
import numpy as np
from ultralytics import YOLO
model=YOLO('yolov8n-pose.pt')
from collections import Counter

class find_jersey_color:
    def __init__(self):
        pass
    
    def find_color(self,image,all_bbox, threshold): 
        color_bbox=[]
        for bbox in all_bbox:
            # print("************************ : ",bbox)
            xmin, ymin, xmax, ymax = int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3])
            each_bbox_img = image[ymin:ymax, xmin:xmax]  # cropp image
            five_points=[]
            center_x = each_bbox_img.shape[1]  // 2
            center_y = each_bbox_img.shape[0] // 2
            five_points.append((center_x,center_y))
            five_points.append((center_x,center_y-10))
            # five_points.append((center_x-5,center_y-5))
            # five_points.append((center_x+5,center_y-5))

            # res_=model.predict(each_bbox_img,save=False,conf=0.25)[0]
            # keypoints = res_.keypoints.data[0].tolist()
            # # print("kkkkkkkkkkkkkk  : ",keypoints)
            # for i, kp in enumerate(keypoints):
            #     if(i==5):
            #         five_points.append((int(kp[0]),int(kp[1])))
            #     if(i==6):
            #         five_points.append((int(kp[0]),int(kp[1])))
            #     if(i==11):
            #         five_points.append((int(kp[0]),int(kp[1])))
            #     if(i==12):
            #         five_points.append((int(kp[0]),int(kp[1])))

            c_bbox,image = self.find_color_on_each_bbox(image,each_bbox_img,bbox,five_points, threshold)
            color_bbox.append(c_bbox)
        # print("len : ",len(five_points),"   : ",five_points)
        # cv2.imshow("mask",image)
        # cv2.waitKey(200)
        # cv2.destroyAllWindows()
        return color_bbox, image


    def find_color_on_each_bbox(self,image,each_bbox_img,bbox,five_points,colors, threshold=0.0):   
        color_dict = {
            'white': [[0, 0, 200], [180, 30, 255]],
            'red': [[0, 100, 100], [10, 255, 255], [160, 100, 100], [180, 255, 255]],
            'yellow': [[20, 100, 100], [40, 255, 255]],
            'green': [[40, 50, 50], [80, 255, 255]],
            'blue': [[100, 50, 50], [130, 255, 255]],
            'light blue': [[90, 100, 50], [125, 255, 180]],
            'dark blue': [[100, 50, 50], [130, 255, 255]],
            'purple': [[130, 50, 50], [150, 255, 255]],
            'pink': [[140, 50, 50], [170, 255, 255]],
            'orange': [[10, 100, 100], [25, 255, 255]],
            'black': [[0, 0, 0], [180, 40, 30]],
            'grey': [[0, 0, 50], [180, 20, 180]],
            'brown': [[0, 50, 50], [30, 255, 150]],
            'beige': [[20, 30, 100], [40, 80, 200]],
            'cyan': [[80, 100, 100], [100, 255, 255]],
            'magenta': [[140, 50, 50], [160, 255, 255]],
            'olive': [[40, 50, 50], [70, 255, 150]],
            'maroon': [[0, 50, 50], [10, 255, 150]],
            'navy': [[100, 50, 50], [130, 255, 150]],
            'teal': [[80, 50, 50], [100, 255, 150]],
            'coral': [[0, 100, 50], [10, 255, 255]],
            'lavender': [[130, 30, 30], [150, 255, 200]],
            # 'silver','','','','','','',
            'gold': [[20, 100, 50], [30, 255, 255]],
            'silver': [[0, 0, 75], [180, 20, 200]],
            'turquoise': [[70, 50, 50], [90, 255, 255]],
            'indigo': [[110, 50, 50], [130, 255, 255]],
            'azure': [[80, 50, 50], [100, 255, 255]],
            'emerald': [[60, 50, 50], [80, 255, 255]],
            'peach': [[0, 50, 80], [20, 255, 255]],
            'rose': [[140, 50, 50], [160, 255, 255]],
            'burgundy': [[0, 50, 20], [10, 255, 80]],
            'lime': [[50, 50, 50], [70, 255, 255]],
            'khaki': [[30, 20, 100], [50, 80, 200]],
            'turquoise blue': [[170, 50, 50], [190, 255, 255]],
            'peacock blue': [[150, 50, 50], [170, 255, 255]],
            'burnt orange': [[10, 100, 50], [20, 255, 255]],
            'dark green': [[50, 50, 50], [80, 255, 150]],
            'dark red': [[0, 50, 50], [10, 255, 150]],
            'dark purple': [[130, 50, 50], [160, 255, 150]],
            'navy blue': [[100, 50, 50], [130, 255, 150]],
            'dark teal': [[80, 50, 50], [100, 255, 150]],
            'dark brown': [[0, 50, 20], [20, 255, 80]],
            'mustard yellow': [[30, 100, 50], [40, 255, 255]],
            'light green': [[40, 50, 50], [80, 255, 180]],
            'light red': [[0, 100, 100], [10, 255, 255], [160, 100, 100], [180, 255, 180]],
            'light purple': [[130, 50, 50], [160, 255, 180]],
            'light pink': [[140, 50, 50], [170, 255, 180]],
            'light orange': [[10, 100, 100], [25, 255, 255]],
            'light grey': [[0, 0, 100], [180, 10, 180]],
            'light brown': [[0, 50, 20], [20, 255, 180]],
            'light yellow': [[20, 100, 100], [40, 255, 255]],
        }

        red_bbox=[]
        xmin, ymin, xmax, ymax = int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3])
        # print("xmin, ymin, width, height : ",xmin, ymin, width, height)
        # Draw a rectangle on the image
        color = (0, 255, 0)  # Green color in BGR format
        thickness = 2  # Thickness of the rectangle border
        # Define the size of the region around the center point that you want to crop
        crop_size = 5  # Adjust this size based on your requirements
        for point_ in five_points:

            center_x = point_[0]
            center_y = point_[1]
            # Calculate the new xmin, ymin, xmax, ymax for the cropped region
            crop_xmin = max(0, center_x - crop_size)
            crop_ymin = max(0, center_y - crop_size )
            crop_xmax = min(each_bbox_img.shape[1], center_x + crop_size )
            crop_ymax = min(each_bbox_img.shape[0], center_y + crop_size )
            
            if crop_xmin > crop_xmax and crop_ymin > crop_ymax:
                continue

            roi = each_bbox_img[crop_ymin:crop_ymax, crop_xmin:crop_xmax]       # detection
            # model=YOLO('yolov8n-pose.pt')
            # res_=model.predict(each_bbox_img,save=False,conf=0.5)
            # cv2.imshow('roi Image', res_[0].plot())
            # cv2.waitKey(200)
            # cv2.destroyAllWindows()

            # Get the dimensions of the image
            height, width, _ = roi.shape
            # roi_hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
            roi = np.clip(roi, 0, 255).astype(np.uint8)

            total_pixels = roi.shape[0] * roi.shape[1]
            # print("roi.shape[0] * roi.shape[1]   :  ",roi.shape[0] , roi.shape[1])
            highest_color = None
            highest_perc = 0
            # print("total_pixels total_pixels : ",total_pixels)
            if total_pixels>0 :
                # Create a mask for the region of interest
                roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            else:
                continue
            
            colors=['red','blue','light blue','dark blue','white','yellow','green','purple','pink','orange','black',
           'grey','brown','beige','cyan','magenta','olive','maroon','navy','gold',]
            for color in colors:
                if color in color_dict.keys():
                    # print(" color from list : ",color)
                    # red is on the edge of HSV range, so needs a combination of two masks
                    if color == 'red':
                        lower1 = np.array(color_dict[color][0])
                        upper1 = np.array(color_dict[color][1])
                        lower2 = np.array(color_dict[color][2])
                        upper2 = np.array(color_dict[color][3])
                        mask = cv2.inRange(roi_hsv, lower1, upper1) + cv2.inRange(roi_hsv, lower2, upper2)
                    else:
                        lower = np.array(color_dict[color][0])
                        upper = np.array(color_dict[color][1])
                        mask = cv2.inRange(roi_hsv, lower, upper)
                    # print("mask   mask   mask  :", mask)
                    color_perc = cv2.countNonZero(mask) / total_pixels
                    # print("color_perc   color_perc   HSV  :", color_perc)
                    if color_perc > highest_perc and color_perc > threshold:
                        # result = cv2.bitwise_and(result, result, mask=mask)
                        highest_perc = color_perc
                        highest_color = color
                        red_bbox.append(highest_color)
                        break
                else:
                    print(f'{color} not in color dictionary')
                    break

            radius = 5
            center=(xmin+center_x,ymin+center_y)
            color_circle=(0, 0, 255)
            # Draw the circle on the image
            cv2.circle(image, center, radius, color_circle, thickness=cv2.FILLED)

        # Use Counter to count the occurrences of each color
        if(len(red_bbox)>0):
            color_counts = Counter(red_bbox)
            # Find the color with the maximum occurrence
            max_color = max(color_counts, key=color_counts.get)
        else:
            max_color="unidentified"
        position=(xmin, ymin)
        font=cv2.FONT_HERSHEY_SIMPLEX
        font_scale=1
        color_text=(0, 255, 0)
        thickness=2
        cv2.putText(image, " "+max_color, position, font, font_scale, color_text, thickness, cv2.LINE_AA)
        
        # output.write(image)

        return max_color,image
        # if highest_color:
        #     return red_bbox ,image
        # else:
        #     return red_bbox,image
    
