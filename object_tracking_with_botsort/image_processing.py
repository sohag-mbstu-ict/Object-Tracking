
import numpy as np
import cv2
class image_processing_techniques:
    def __init__(self):
        pass

    def background_subtr_opencv(self,image_,bg_subtractor):
        # Apply background subtraction
        fg_mask = bg_subtractor.apply(image_)
        #Threshold the mask to extract moving objects
        threshold = 88
        fg_mask_binary = cv2.threshold(fg_mask, threshold, 255, cv2.THRESH_BINARY)[1]
        # Apply the mask to the original frame to get the moving objects
        image_ = cv2.bitwise_and(image_, image_, mask=fg_mask_binary)
        return image_
    
    
    def background_Substitution(self,im0,background_subtractor):
        # Apply background subtraction
        fg_mask = background_subtractor.apply(im0)

        # Optional: Smooth the mask using morphological operations
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, (5, 5), iterations=2)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, (5, 5), iterations=2)
        # Apply the mask to the original frame to get the foreground
        fg_result = cv2.bitwise_and(im0, im0, mask=fg_mask)
        return fg_result
        


    def sharpen_image(self,image):
        kernel = np.array([[-1, -1, -1],
                        [-1,  9, -1],
                        [-1, -1, -1]])
        sharpened_image = cv2.filter2D(image, -1, kernel)
        return sharpened_image


    #Histogram Equalization:
    def enhance_contrast(self,image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        equalized = cv2.equalizeHist(gray)
        return cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)



    #Reduce Noise using Gaussian Blur
    def reduce_noise(self,image):
        return cv2.GaussianBlur(image, (5, 5), 0)



    #Normalize Pixel Values to [0, 1]
    def normalize_image(self,image):
        return image.astype(np.float32) / 255.0



    def super_resolution(self,image):
        # Define the scale factor for super-resolution (e.g., 2x)
        scale_factor = 1

        # Upscale the image using cubic interpolation
        upscaled_image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
        return upscaled_image



    def color_Enhancement(self,image):
        # Convert the image to LAB color space
        lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        # Split the LAB image into channels
        l_channel, a_channel, b_channel = cv2.split(lab_image)
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to the L channel
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l_channel_clahe = clahe.apply(l_channel)
        # Merge the enhanced L channel with the original A and B channels
        enhanced_lab_image = cv2.merge((l_channel_clahe, a_channel, b_channel))
        # Convert the enhanced LAB image back to BGR color space
        enhanced_image = cv2.cvtColor(enhanced_lab_image, cv2.COLOR_LAB2BGR)
        return enhanced_image
    
    #Convert the image to grayscale. This eliminates color information but retains intensity, making it less sensitive to color variations
    def grayscale_conversion(self,image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Create a background subtractor
        bg_subtractor = cv2.createBackgroundSubtractorMOG2()
        # Apply background subtraction
        fg_mask = bg_subtractor.apply(gray_image)
        #Threshold the mask to extract moving objects
        threshold = 88
        fg_mask_binary = cv2.threshold(fg_mask, threshold, 255, cv2.THRESH_BINARY)[1]
        # Apply the mask to the original frame to get the moving objects
        image_ = cv2.bitwise_and(gray_image, gray_image, mask=fg_mask_binary)
        return image_
    

    def get_video_info(self,video_path):
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        # Check if the video file is opened successfully
        if not cap.isOpened():
            raise ValueError("Error: Could not open the video file.")
        # Get the frame width and frame height
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return cap, frame_width, frame_height
    

    def make_video_from_tracker(self,fg_result,output,make_video,color_bbox):
        # Open the video file
        # print("inside img tech : ",output.T.base)
        c=0
        for box in output.T.base:
            xmin, ymin,xmax,ymax=int(box[0]), int(box[1]), int(box[2]), int(box[3])
            id_name=int(box[4])
            color = (0, 255, 0)  # Green color in BGR format
            thickness = 2  # Thickness of the rectangle border
            cv2.rectangle(fg_result, (xmin, ymin), (xmax, ymax), color, thickness)
            color = (0, 0, 0)  # Green color in BGR format
            thickness = 2  # Thickness of the rectangle border
            font = cv2.FONT_HERSHEY_SIMPLEX  # Font style
            font_scale = 0.8  # Font scale
            font_thickness = 2  # Thickness of the font stroke
            text_str=""
            track_id_text=text_str +"_"+ str(id_name)
            # Write text on top of the bounding box
            cv2.putText(fg_result, str(id_name), (xmin, ymin - 5), font, font_scale, color, font_thickness)

        make_video.write(fg_result)
        # Display the image with the rectangle
        # cv2.imshow("l",fg_result)
        # cv2.waitKey(200)
        # cv2.destroyAllWindows()
        


    

