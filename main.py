import cv2
import numpy as np

import argparse
import glob
import os

from model.classifier import Classifier
from tracker import tracker, tracker_args
from utils.helpers import get_depth, draw_text


def get_args():
    parser = argparse.ArgumentParser(description='Something something')

    parser.add_argument('--path',
                      type=str,
                      help= 'Path to the image sequence')
    parser.add_argument('--debug',
                        type=str,
                        help='Toggle debug mode',
                        default=False)

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    print(args)

    # image_list_left = glob.glob(os.path.join(args.path, 'image_2/data/*.png'))
    # image_list_right = glob.glob(os.path.join(args.path, 'image_3/data/*.png'))
    image_list_left = glob.glob(os.path.normpath('raw_videos/seq_03/image_02/data/*.png'))
    image_list_right = glob.glob(os.path.normpath('raw_videos/seq_03/image_03/data/*.png'))

    # Load model and tracker
    classifier = Classifier(train_method='coco', format='pt')
    byte_tracker = tracker.BYTETracker(tracker_args.BYTETrackerArgs())

    CLASS_NAMES_DICT = classifier.model.names

    result_video = []

    for image_file_left, image_file_right in zip(image_list_left, image_list_right):     
        image_left = cv2.imread(image_file_left)
        image_right = cv2.imread(image_file_right)
        overlay = image_left.copy()

        # Apply object detection to the image and classify the object use image in RGB format
        results = classifier.predict(image_left[:, :, ::-1])[0].cpu()

        if len(results.boxes) >= 2:
            
            # Compute the depth map 
            depth = get_depth(image_left, image_right)
            
            # Track the position by applying Kalman Filter
            updated_result = byte_tracker.update(results.boxes, depth)
            
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
                    image_left = cv2.circle(np.array(image_left), (int((bbox[0] +bbox[3])/2), int((bbox[1] + bbox[4])/2)), 0, (255, 0, 155), 20)
                    cv2.rectangle(overlay, (int(bbox[0]), int(bbox[1])), (int(bbox[3]), int(bbox[4])), (0, 0, 255), 2)
                    image_new = draw_text(overlay, text1, text2, pos=(int(bbox[0]), int(bbox[1])), font_scale=0.4, alpha=0.6)

        h, w, _ = image_new.shape
        
        # Add new frame to video
        result_video.append(image_new[:,:,::-1])
        
        if args.debug:
            while True:
                key = cv2.waitKey(0) & 0xFF
                if key == ord('n'):
                    break

                
    # Write all processed images into a video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out =cv2.VideoWriter('out.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 3, (w,h)) 

    for i in range(len(result_video)):
        out.write(result_video[i][:, :, ::-1])
        
    out.release()    

