from classification import Classifier
from dataclasses import dataclass
from my_byte_tracker import BYTETracker
from ultralytics import YOLO

import cv2
import glob
import numpy as np

@dataclass(frozen=True)
class BYTETrackerArgs:
    track_low_thresh = 0.2
    track_high_thresh = 0.6
    new_track_thresh: float = 0.15
    track_buffer: int = 300
    match_thresh: float = 0.9
    aspect_ratio_thresh: float = 3.0
    min_box_area: float = 1.0
    mot20: bool = False
    
def get_depth(imgL, imgR):
    imgL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    imgR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
    
    # focal_length = K_02[0][0]
    focal_length = 956.9475
    
    # b = np.linalg.norm(t_02-t_03)
    b = np.linalg.norm(np.array([0.059896, -0.001367835, 0.004637624 ])-np.array([-0.4756270, 0.005296617, -0.005437198]))
    
    stereo = cv2.StereoBM_create()
    stereo.setMinDisparity(4)
    stereo.setNumDisparities(128)
    stereo.setBlockSize(21)
    stereo.setSpeckleRange(16)
    stereo.setSpeckleWindowSize(45)
    
    disparity = stereo.compute(imgL,imgR)
    disparity[disparity <=0] = 1e-5
   
    depth = (b * focal_length)  / disparity
        
    return depth

def draw_text(img, text1, text2,
          font=cv2.FONT_HERSHEY_SIMPLEX,
          pos=(0, 0),
          font_scale=0.3,
          font_thickness=1,
          text_color=(255, 255, 255),
          text_color_bg=(0,0,255)
          ):
    x, y = pos
    text_size, _ = cv2.getTextSize(text2, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv2.rectangle(img, pos, (x + text_w, y + 2*text_h), text_color_bg, -1)
    cv2.putText(img, text1, (x,  y + text_h), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
    cv2.putText(img, text2, (x,  y + 2*text_h), font, font_scale, text_color, font_thickness, cv2.LINE_AA)

    return text_size


if __name__ == '__main__':
    # Load images
    img_left = sorted(glob.glob('final_project_2023_rect/seq_03/image_02/data/*.png'))
    img_right = sorted(glob.glob('final_project_2023_rect/seq_03/image_03/data/*.png'))

    # Load the model and the tracker
    classifier = Classifier(train_method="coco", format='pt')
    byte_tracker = BYTETracker(BYTETrackerArgs())
    model = YOLO('weights/coco.pt')

    CLASS_NAMES_DICT = model.model.names

    result_video = []
    
    for i in range(0, len(img_left)):
        frame_left = cv2.imread(img_left[i])
        frame_right =  cv2.imread(img_right[i])
        overlay = frame_left.copy()
        
        # Object detection and classification
        predicted_result = classifier.predict(frame_left)
        
        if len(predicted_result[0].boxes) >= 2:
            
            # Compute the depth map 
            depth = get_depth(frame_left, frame_right)
            
            # Track the position by applying Kalman Filter
            updated_result = byte_tracker.update(predicted_result[0].boxes, depth)
            
            # Draw boxes for visualization
            if len(updated_result) > 0:
                
                for j, (result) in enumerate(updated_result):
                    
                    # Get parameters
                    bbox = result[0:5]
                    id = result[5]
                    score = result[6]
                    cls = result[7]
                    
                    text1 =  str(int(id))+ ", " + CLASS_NAMES_DICT[cls]
                    text2 = f"{bbox[0]:.2f}, " + f"{bbox[1]:.2f}, " + f"{bbox[2]:.2f}"
                    
                    # Draw the center of the object, the rectangle and add text
                    frame_left = cv2.circle(frame_left, (int((bbox[0] +bbox[3])/2), int((bbox[1] + bbox[4])/2)), 0, (255, 0, 155), 20)
                    cv2.rectangle(overlay, (int(bbox[0]), int(bbox[1])), (int(bbox[3]), int(bbox[4])), (0, 0, 255), 2)
                    draw_text(overlay, text1, text2, pos=(int(bbox[0]), int(bbox[1])), font_scale=0.4)
                    
                    # Transparency factor.
                    alpha = 0.6  
                    # Following line overlays transparent rectangle over the image
                    image_new = cv2.addWeighted(overlay, alpha, frame_left, 1 - alpha, 0)
                  
        h, w, _ = image_new.shape
        
        # Add new frame to video
        result_video.append(image_new)
        #cv2.imshow('Frame', image_new)
        #cv2.waitKey(500)
            
    # cv2.destroyAllWindows() 
    
    # Write all processed images into a video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out =cv2.VideoWriter('out.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 3, (w,h)) 

    for i in range(len(result_video)):
        out.write(result_video[i])
        
    out.release()
                    