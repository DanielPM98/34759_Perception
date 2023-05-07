import numpy as np

class KalmanFilter(object):
    
    """docstring for KalmanFilter"""
 
    def __init__(self ):
        #super(KalmanFilter, self).__init__()
        self.initModel()
	
    """init function to initialise the model"""
    def initModel(self): 
		# The initial state (6x1).
         self.x = np.array([[0], # Position along the x-axis
                    [0], # Velocity along the x-axis
                    [0], # Position along the y-axis
                    [0], # Velocity along the y-axis
                    [0], # Position along the z-axis
                    [0]])# Velocity along the z-axis

        # The initial uncertainty (6x6).
         self.P = np.array([[300, 0, 0, 0, 0, 0],
                    [0, 300, 0, 0, 0, 0],
                    [0, 0, 300, 0, 0, 0],
                    [0, 0, 0, 300, 0, 0],
                    [0, 0, 0, 0, 300, 0],
                    [0, 0, 0, 0, 0, 300]])

        # The external motion (6x1).
         self.u = np.array([[0],
                    [0],
                    [0],
                    [0],
                    [0],
                    [0]])

        # The transition matrix (6x6).  
         self.F = np.array([[1, 1, 0, 0, 0.0, 0],
                            [0, 1, 0, 0, 0, 0],
                            [0, 0, 1, 1, 0, 0],
                            [0, 0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 1, 1],
                            [0, 0, 0 ,0, 0, 1]])

        # The observation matrix (2x6).
         self.H = np.array([[1, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0]])

        # The measurement uncertainty.
         self.R = 0.75*1
                    
        # The identity matrix. Simply a matrix with 1 in the diagonal and 0 elsewhere.
         self.I = np.array([[1, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 1]])
    
    """Predict function which predicst next state based on previous state"""
    def predict(self):
        ### insert predict function
        self.x = np.dot(self.F, self.x) + self.u
        self.P = np.dot(np.dot(self.F, self.P), np.transpose(self.F))
        
        return self.x

    """Correct function which correct the states based on measurements"""
    def update(self, Z):
        ### Insert update function
        Y = Z - np.dot(self.H, self.x)
        S = np.dot(np.dot(self.H, self.P), np.transpose(self.H)) + self.R
        K = np.dot(np.dot(self.P, np.transpose(self.H)), np.linalg.pinv(S))
        self.x = self.x + np.dot(K, Y)
        self.P = np.dot((self.I - np.dot(K, self.H)), self.P)
        
        return self.x