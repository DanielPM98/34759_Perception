from ultralytics import YOLO

class Classifier:
    def __init__(self, train_method = 'own', format = 'pt'):
        if train_method == 'own':
            self.weights = 'weights/best_run.' + format # Trained network
        elif train_method == 'coco':
            self.weights = 'weights/coco.' + format # Coco trained
        else:
            self.weights = 'weights/yolo8n.' + format # Standard weights for detection training

        self.model = YOLO(self.weights, task='detect')

    def predict(self, image):
        results = self.model(image)
        
        return results

