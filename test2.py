from my_byte_tracker import BYTETracker, STrack
from onemetric.cv.utils.iou import box_iou_batch
from dataclasses import dataclass
    
from ultralytics import YOLO
import glob
import cv2
from classification import Classifier
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
     # min_disp = 1
    # stereo.setMinDisparity(min_disp)
    #stereo.setMinDisparity(2)
    # stereo.setDisp12MaxDiff(100)
    # stereo.setUniquenessRatio(50)
    # stereo.setSpeckleRange(3)
    # stereo.setSpeckleWindowSize(50)
    disparity = stereo.compute(imgL,imgR)
    disparity[disparity <=0] = 1e-5
    norm_image = cv2.normalize(disparity, None, alpha = 0, beta = 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    dst = cv2.GaussianBlur(disparity,(7,7),cv2.BORDER_DEFAULT)
    depth = (b * focal_length)  / disparity
    
    return depth


img_left = sorted(glob.glob('final_project_2023_rect/seq_02/image_02/data/*.png'))
img_right = sorted(glob.glob('final_project_2023_rect/seq_02/image_03/data/*.png'))


classifier = Classifier(train_method="coco", format='pt')
model = YOLO('weights/coco.pt')

byte_tracker = BYTETracker(BYTETrackerArgs())
for i in range(0, len(img_left)):
    frame = cv2.imread(img_left[i])
    frame_right =  cv2.imread(img_right[i])
    
    depth = get_depth(frame, frame_right)
    #cv2.imshow('Frame', depth)
    #cv2.waitKey(10000)
    
    results = classifier.predict(frame)
    
    
    for i in range(0,len(results[0])):
        # print(results[0].boxes[i])
        box = results[0].boxes[i].xywh
    
    outputs = byte_tracker.update(results[0].boxes, depth)
    tracks_boxes = np.array([track for track in outputs], dtype=float)
    #iou = box_iou_batch(tracks_boxes, results[0].boxes.xyxy)
    
    # track2detection = np.argmax(iou, axis=1)
    
    # tracker_ids = [None] * len(results[0].boxes)
    
    # for tracker_index, detection_index in enumerate(track2detection):
    #     if iou[tracker_index, detection_index] != 0:
    #         tracker_ids[detection_index] = outputs[tracker_index].track_id

    
    # draw boxes for visualization
    if len(outputs) > 0:
        
        
        for j, (output) in enumerate(outputs):
            
            print(output)
            bbox = output[0:5]
            id = output[5]
            cls = output[6]
            conf = output[7]
            
            frame = cv2.circle(frame, (int((bbox[0] +bbox[3])/2), int((bbox[1] + bbox[4])/2)), 0, (255, 0, 155), 20)
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[3]), int(bbox[4])), (0, 0, 255), 2)
            
            frame = cv2.putText(frame, str(id) + " " + str(bbox[2]), (int(bbox[0]), int(bbox[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1, cv2.LINE_AA)
    cv2.imshow('Frame', frame)
    cv2.waitKey(500)
    
cv2.destroyAllWindows()  

            