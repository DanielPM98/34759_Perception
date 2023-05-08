import cv2
import glob
import numpy as np
import torch

from model.classifier import Classifier
from Tracker_fail_2 import Tracks
from scipy.optimize import linear_sum_assignment

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

def main():
    # List for  tracking objects
    trackers = []
    for i in range(0, len(images_left)):
        frame_left  = cv2.imread(images_left[i])
        frame_right = cv2.imread(images_right[i])
        
       # Object detection and classification
        results = classifier.predict(frame_left)

        cens = []
        dim = []
        types = []

        for i in range(0,len(results[0])):
            
            # Get box parameters
            box = results[0].boxes[i].xywh
            box = torch.squeeze(box)
            types.append(results[0].boxes[i].cls)
            
            # Get only objects detected as people
            if (results[0].boxes[i].cls == 0):
                cens.append(np.asarray([box[0], box[1]]))
                dim.append(np.asarray([box[2], box[3]]))
        
        thresh_value = 70
        # Add objects to be tracked
        for i in range(len(cens)):
            cen = np.array(cens[i], dtype=np.float32)
            min = 1000
            
            # Compute the distance between objects
            for tracker in trackers:
                if(np.linalg.norm(tracker.centroid - cen) < min):
                    min = np.linalg.norm(tracker.centroid - cen)
            
            # Add one object in the list only if it is not too close of another 
            if (min > thresh_value):
                # Get the depth map
                depth_map = get_depth(frame_left, frame_right)
                
                # Get correspondent depth of the center of the object
                z_obj = depth_map[int(cens[i][1])][int(cens[i][0])]
                
                # Create a new Tracks object
                track = Tracks(i, cens[i], z_obj, dim[i][1], dim[i][0])
                
                Z = np.array([[cens[i][0]],[cens[i][1]], [z_obj]])
                
                # Update new position using Kalman filter
                x = track.kalman.update(Z)
                trackers.append(track)
 
        # Predict new location for all existing trackers
        for tracker in trackers:
            x = tracker.kalman.predict()
            
            # Update the centroid and the depth
            tracker.centroid = (int(x[0]), int(x[2]))
            tracker.depth = (int(x[4]))
            
        # Tracker management
        # Calculate lose
        costs = []
        for tracker in trackers:
            current_costs = []
            for c in cens:
                cen = np.array(c, dtype=np.float32)
                current_costs.append(np.linalg.norm(tracker.centroid - cen))
            costs.append(current_costs)
        
        
        _to_delete = []
        if (len(trackers) >= 2):
            
            # Assign tracker to object, update kalman filter,
            tracker_id, obj_id = linear_sum_assignment(np.array(costs, dtype=np.float32))

            for tid, oid in zip(tracker_id, obj_id):
                measure_cen = np.array(cens[oid])
                dist = np.linalg.norm(trackers[tid].centroid - measure_cen)
                
                if (dist < thresh_value):
                    Z = np.array([[measure_cen[0]],[measure_cen[1]], [trackers[tid].depth]])
                    x = trackers[tid].kalman.update(Z)
                    
                else:
                    _to_delete.append(trackers[tid])
            
        for i in range(len(trackers)):
            
            res = (i in tracker_id)
            if (res == False):
                trackers[i].__set_consec_frames__(trackers[i].__get_consec_frames__() + 1)
            else:
                trackers[i].__set_consec_frames__(0)

            
            if (trackers[i].__get_consec_frames__() > 10):
                 _to_delete.append(trackers[i])

        if (len(_to_delete) > 0):
            trackers = [tracker for tracker in trackers if tracker not in _to_delete]
        
        for tid in range(len(trackers)):
            cx = int(trackers[tid].centroid[0])
            cy = int(trackers[tid].centroid[1])
            depth = trackers[tid].depth

            frame_left = cv2.circle(frame_left, (int(cx), int(cy)), 0, (255, 0, 155), 20)
            cv2.rectangle(frame_left, (int(cx - trackers[tid].width/2), int(cy - trackers[tid].height/2)), (int(cx + trackers[tid].width//2), int(cy + trackers[tid].height/2)), (0, 0, 255), 2)
            
        text1 = "Position x= {}, y ={} z={}".format(x[0], x[2], depth)
                                    
        cv2.putText(frame_left, text1, (10, 50),  cv2.FONT_HERSHEY_DUPLEX, 0.4, (255, 255, 255), 2)  
        for tracker in cens:
            cx = int(tracker[0])
            cy = int(tracker[1])
            frame_left = cv2.circle(frame_left, (int(cx), int(cy)), 0, (0, 155, 155), 20)

        cv2.imshow('Frame', frame_left)
        cv2.waitKey(500)

    cv2.destroyAllWindows()

        
if __name__ == '__main__':

    # Load the images
    images_left = sorted(glob.glob('final_project_2023_rect/seq_01/image_02/data/*.png'))
    images_right = sorted(glob.glob('final_project_2023_rect/seq_01/image_03/data/*.png'))
    assert images_left
    assert images_right
    
    classifier = Classifier(train_method="coco", format='pt')
    main()