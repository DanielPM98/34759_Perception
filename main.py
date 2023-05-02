import cv2
import torch

import argparse
import glob
import os

from classification import Classifier
from tracker import KalmanTracker

def get_args():
    parser = argparse.ArgumentParser(description='Something something')

    parser.add_argument('--path',
                      type=str,
                      help= 'Path to the image directory')
    parser.add_argument('--debug',
                        type=str,
                        help='Toggle debug mode',
                        default=False)

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    image_list = glob.glob(os.path.join(args.path, '*.png'))
    classifier = Classifier(pre_train=True, format='pt')
    tracker = KalmanTracker(dt=1/30, ux=1, uy=1, std_acc=1, std_x=0.1, std_y=0.1 )

    for image_file in image_list:
        # img_path = os.path.join(args.path, image_file)
        
        # Load image from directory and parse to RGB
        image = cv2.imread(image_file)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply object detection to the image and classify the object
        results = classifier.predict(image_rgb) # TODO: Define what output we want

        box = results[0].boxes[0].xywh
        box = torch.squeeze(box)




        cv2.circle(image, (int(box[0]), int(box[1])), 3, (0, 0, 255), 5)
        cv2.imshow('Testing image', image)
        
        
        
        if args.debug:
            while True:
                key = cv2.waitKey(0) & 0xFF
                if key == ord('n'):
                    break
        else:
            cv2.waitKey(50)
        
        # Apply 3D tracking 

    cv2.destroyAllWindows()