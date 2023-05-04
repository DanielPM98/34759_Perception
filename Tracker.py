from Kalman_filter import KalmanFilter

class Tracks(object):
    def __init__(self, id, centroid, depth, height, width):
        self.id = id
        self.centroid = centroid
        self.depth = depth
        self.kalman = KalmanFilter()
        self.height = height
        self.width = width
        self.consecutive_frames = 0
        
    def __get_consec_frames__(self):
        return self.consecutive_frames
    
    def __set_consec_frames__(self, frames):
        self.consecutive_frames = frames
        

        