
import cv2
import pandas as pd
import json
result_file = "E:/dataset/run/detections.json"
class store_bbox_id_number_in_json_csv_file:
    def __init__(self):
        self.frame_count=0
        self.df=pd.DataFrame(columns=['bbox','tracking_id'])


    def store_bbox_id_number(self,c,boxes,track_ids):
        dt_total_list = []
        detections = []
        for i in range(len(boxes)):
            # print("box : ,track_ids   - ",boxes[i],track_ids[i])
            bbox=boxes[i].tolist()
            bbox=[round(num, 4) for num in bbox]
            detections.append({
                "fr":c,
                "bbox": bbox,
                "track_id": track_ids[i],
            })

        # print("detections : ",detections)
        if result_file:
            with open(result_file, "a") as f:
                json.dump(detections, f)
        # return {"Batch_16": detections}
                
                
    def store_bbox_tracking_id_number_in_csv(self,tracks,df,indx):
        boxes = tracks[0].boxes.xyxy.cpu()
        # boxes = tracks[0].boxes.xyxy.cpu().tolist()
        clss = tracks[0].boxes.cls.cpu().tolist()
        
        track_ids = tracks[0].boxes.id.int().cpu().tolist()
        # print("Inside object-ounter boxes,track_ids : ",boxes,track_ids)
        # obj_json.store_bbox_id_number(boxes,track_ids) # call the function to store bbox, ID in json
        # bbox_df=boxes.tolist()
        if(len(track_ids)>1):
            bbox_df = [[round(num, 2) for num in box] for box in boxes.tolist()]
            df.loc[indx]=[bbox_df,track_ids] # store frame_num,bbox,tracking id in csv file 
        else:
            bbox_df = [[round(num, 2) for num in box] for box in boxes.tolist()]
            df.loc[indx]=[bbox_df,None]

        return df



