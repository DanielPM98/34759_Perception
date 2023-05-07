from motpy import Detection, MultiObjectTracker
import numpy as np
import cv2
import torch

import argparse
import glob
import os

from classification import Classifier
from utils import draw_boxes

PALETTE = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 0, 0), ]
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

    # image_list_left = glob.glob(os.path.join(args.path, 'image_2/data/*.png'))
    # image_list_right = glob.glob(os.path.join(args.path, 'image_3/data/*.png'))
    image_list_left = glob.glob(os.path.normpath('raw_videos/seq_03/image_02/data/*.png'))
    image_list_right = glob.glob(os.path.normpath('raw_videos/seq_03/image_03/data/*.png'))

    # Load model and tracker
    classifier = Classifier(train_method='coco', format='pt')
    tracker = MultiObjectTracker(dt=0.1)

    CLASS_NAMES_DICT = classifier.model.names

    for image_file_left, image_file_right in zip(image_list_left, image_list_right):     
        # Load image from directory and parse to RGB
        image_left = cv2.imread(image_file_left)[:, :, ::-1]
        image_right = cv2.imread(image_file_right)[:, :, ::-1]
        overlay = image_left.copy()

        # Apply object detection to the image and classify the object
        results = classifier.predict(image_left)[0]

        object_boxes = []
        for box in results.boxes:
            # print(box)
            box = torch.squeeze(box.xyxy).cpu().numpy()
            object_box = Detection(box=box)
            # print('Second box')
            # print(box)
            object_boxes.append(object_box)


        # update the state of the multi-object-tracker tracker
        # with the list of bounding boxes
        tracker.step(detections=object_boxes)

        # retrieve the active tracks from the tracker (you can customize
        # the hyperparameters of tracks filtering by passing extra arguments)
        active_tracks = tracks = tracker.active_tracks()

        image_new = draw_boxes(overlay, active_tracks, alpha=0.6)

        
        if args.debug:
            while True:
                key = cv2.waitKey(0) & 0xFF
                if key == ord('n'):
                    break
        else:
            cv2.waitKey(50)
        
        # Apply 3D tracking 

    cv2.destroyAllWindows()