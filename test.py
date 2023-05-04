import cv2
import numpy as np
import imutils
import glob
import torch
from classification import Classifier
from Tracker import Tracks
from scipy.optimize import linear_sum_assignment # to assign detectors to objects 

def test():
    trackers = []
    for i in range(0, 50): # len(images_left)):
        frame = cv2.imread(images_left[i])
        frame_right =  cv2.imread(images_right[i])
        
        results = classifier.predict(frame)
        detection_num = len(results[0])
        cens = []
        dim = []
        people = []
        types = []

        for i in range(0,len(results[0])):
            box = results[0].boxes[i].xywh
            box = torch.squeeze(box)
            #cens.append(np.asarray([box[0], box[1]]))
            types.append(results[0].boxes[i].cls)
            
            if (results[0].boxes[i].cls == 0):
                cens.append(np.asarray([box[0], box[1]]))
                dim.append(np.asarray([box[2], box[3]]))
        
        if (len(trackers) == 0):
            for i in range(len(cens)):
                cen = np.array(cens[i], dtype=np.float32)
                min = 1000
                for tracker in trackers:
                    if(np.linalg.norm(tracker.centroid - cen) < min):
                        min = np.linalg.norm(tracker.centroid - cen)
                if (min > 100):
                    z_obj = get_depth(frame, frame_right, int(cens[i][0]), int(cens[i][1]))
                    track = Tracks(i, cens[i], z_obj, dim[i][1], dim[i][0])
                    Z = np.array([[cens[i][0]],[cens[i][1]], [z_obj]])
                    x = track.kalman.update(Z)
                    trackers.append(track)
                
            pedes    = trackers #[track for index,track in enumerate(trackers) if types[index] == 0]
            bicycles = [track for index,track in enumerate(trackers) if types[index] == 1]
            cars     = [track for index,track in enumerate(trackers) if (types[index] == 2 or types[index] == 7)]
            print("Pedes", pedes)
            
        else:
            for i in range(len(cens)):
                cen = np.array(cens[i], dtype=np.float32)
                min = 1000
                for tracker in trackers:
                    if(np.linalg.norm(tracker.centroid - cen) < min):
                        min = np.linalg.norm(tracker.centroid - cen)
                # print(min)
                if (min > 100):
                    z_obj = get_depth(frame, frame_right, int(cens[i][0]), int(cens[i][1]))
                    track = Tracks(len(trackers), cens[i], z_obj, dim[i][1], dim[i][0])
                    Z = np.array([[cens[i][0]],[cens[i][1]], [z_obj]])
                    x = track.kalman.update(Z)
                    trackers.append(track)
                         
        
        # predict new location for all existing tracker
        for tracker in trackers:
            x = tracker.kalman.predict()
            tracker.centroid = (int(x[0]), int(x[2]))
            
        # Tracker management
        # calculate lose
        costs = []
        for tracker in trackers:
            current_costs = []
            for c in cens:
                cen = np.array(c, dtype=np.float32)
                current_costs.append(np.linalg.norm(tracker.centroid - cen)) # prediction and detection
            costs.append(current_costs)
        
        # assign tracker to object, update kalman filter, set invisible number to 0
        tracker_id, obj_id = linear_sum_assignment(np.array(costs, dtype=np.float32))
        print(tracker_id, obj_id)
        for tid, oid in zip(tracker_id, obj_id):
            measure_cen = np.array(cens[oid])
            dist = np.linalg.norm(trackers[tid].centroid - measure_cen)
            if (dist < 40):
                Z = np.array([[measure_cen[0]],[measure_cen[1]], [trackers[tid].depth]])
                x = trackers[tid].kalman.update(Z)
                
            else:
                trackers[tid].__set_consec_frames__(trackers[tid].__get_consec_frames__() + 1)
        
        _to_delete = []
        for i in range(len(trackers)):
            #print(trackers[i].__get_consec_frames__())
            res = i in tracker_id
            if (res == False):
                trackers[i].__set_consec_frames__(trackers[i].__get_consec_frames__() + 1)
            
            if (trackers[i].__get_consec_frames__() > 5):
                 _to_delete.append(trackers[i])

        if (len(_to_delete) > 0):
            trackers = [tracker for tracker in trackers if tracker not in _to_delete]
        
        for tid in range(len(trackers)):
            cx = int(trackers[tid].centroid[0])
            cy = int(trackers[tid].centroid[1])
            
            frame = cv2.circle(frame, (int(cx), int(cy)), 0, (255, 0, 155), 20)
            cv2.rectangle(frame, (int(cx - trackers[tid].width/2), int(cy - trackers[tid].height/2)), (int(cx + trackers[tid].width//2), int(cy + trackers[tid].height/2)), (0, 0, 255), 2)
            
        text1 = "Position x= {}, y ={} z={}".format(x[0], x[2], 0)
        #print(trackers)
        #if (len(people) > 0):
           # trackers = people.copy()
                                              
        cv2.putText(frame, text1, (10, 50),  cv2.FONT_HERSHEY_DUPLEX, 0.4, (255, 255, 255), 2)  
        for tracker in cens:
            cx = int(tracker[0])
            cy = int(tracker[1])
            frame = cv2.circle(frame, (int(cx), int(cy)), 0, (0, 155, 155), 20)
            
        # show prediction
        #for tracker in trackers:
        #    cx = int(tracker.centroid[0])
        #    cy = int(tracker.centroid[1])
        #    frame = cv2.circle(frame, (int(cx), int(cy)), 0, (155, 0, 155), 20)
        #    cv2.rectangle(frame, (int(cx - tracker.width/2), int(cy - tracker.height/2)), (int(cx + tracker.width//2), int(cy + tracker.height/2)), (0, 0, 255), 2)
    
        cv2.imshow('Frame', frame)
        cv2.waitKey(500)
        
    print(len(trackers))
    cv2.destroyAllWindows()
        
def get_depth(imgL, imgR, x, y):
    imgL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    imgR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
    # focal_length = K_02[0][0]
    focal_length = 956.9475
    # b = np.linalg.norm(t_02-t_03)
    b = np.linalg.norm(np.array([0.059896, -0.001367835, 0.004637624 ])-np.array([-0.4756270, 0.005296617, -0.005437198]))
    stereo = cv2.StereoBM_create(numDisparities=8*16, blockSize=15)
    disparity = stereo.compute(imgL,imgR)
    norm_image = cv2.normalize(disparity, None, alpha = 0, beta = 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    dst = cv2.GaussianBlur(norm_image,(7,7),cv2.BORDER_DEFAULT)
    depth = (b * focal_length) / disparity[y][x]
    
    return depth
        
if __name__ == '__main__':
    trackers = []

    pedes = []
    bicycles = []
    cars = []
    # Load the video
    images_left = sorted(glob.glob('final_project_2023_rect/seq_01/image_02/data/*.png'))
    images_right = sorted(glob.glob('final_project_2023_rect/seq_01/image_03/data/*.png'))
    assert images_left
    assert images_right
    classifier = Classifier(train_method="coco", format='pt')
    test()