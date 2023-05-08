import torch
from ultralytics import YOLO

class Classifier:
    def __init__(self, train_method = 'own', format = 'pt'):
        if train_method == 'own':
            self.weights = './model/weights/best_run.' + format # Trained network
        elif train_method == 'coco':
            self.weights = './/model/weights/coco.' + format # Coco trained
        else:
            self.weights = './model/weights/yolo8n.' + format # Standard weights for detection training

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = YOLO(self.weights, task='detect')


    def predict(self, image):
        results = self.model(image, classes=[0, 1, 2])
        
        return results
    

    def train(self, yaml_file, **kwargs):
        results = self.model.train(data=yaml_file, **kwargs)

        return results
    

    def val(self):
        try:
            self.model.val()
        except Exception as e: raise e
