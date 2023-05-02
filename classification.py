from ultralytics import YOLO

class Classifier:
    def __init__(self, pre_train = False, format = 'pt'):
        if pre_train:
            self.weights = 'weights/best_run.' + format # Trained network
        else:
            self.weights = 'weights/yolo8n.' + format

        self.model = YOLO(self.weights, task='detect')

    def predict(self, image):
        results = self.model(image)
        
        return results

