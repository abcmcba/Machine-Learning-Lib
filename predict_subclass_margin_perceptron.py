from cs475_types import ClassificationLabel, FeatureVector, Instance, Predictor
import scipy
import numpy

class predictor_margin_perceptron(Predictor):
    def __init__(self, algorithm, online_training_iterations, online_learning_rate):
        self.eta = online_learning_rate
        self.I = online_training_iterations
        self.algorithm = algorithm
        self.w_hat = []
        
    def train(self, instances):
        maxdim = instances[0]._feature_vector.maxindex
        w = [0] * maxdim      
        N = len(instances)
        wbefore = [0] * maxdim
                                                                      
        for k in range(self.I):
            
            for i in range(N):
                x = [0] * maxdim
                y = instances[i]._label.label     
                if y == 0:
                    y = -1                              
                dictpairs = instances[i]._feature_vector.dict.items()
                for keyindex, valuevalue in dictpairs:
                    x[keyindex-1] = valuevalue
                wdotx = 0
                for j in range(maxdim):     
                    wdotx += w[j] * x[j]

                # update w                                        
                if wdotx * y < 1:
                    valincrease = [self.eta * y * z for z in x]
                    w = [m + n for m, n in zip(w, valincrease)]
                    
        if self.algorithm == "margin_perceptron":
            self.w_hat = w
        else:
            return None

     
    def predict(self, instance):
        dictpairs = instance._feature_vector.dict.items()
        maxdim = instance._feature_vector.maxindex 
        x = [0] * maxdim
        for keyindex, valuevalue in dictpairs:
            x[keyindex-1] = valuevalue
            
        # in case dimension is different
        dimw = len(self.w_hat)
        if dimw == maxdim:                               
            pass
        elif dimw < maxdim:
            self.w_hat += [0] * (maxdim-dimw)
        elif dimw > maxdim:
            self.w_hat = self.w_hat[:maxdim]
            
        wdotx = 0
        for j in range(maxdim):
            wdotx += self.w_hat[j] * x[j]
        yhat = int(numpy.sign(wdotx))
        if yhat == 0:
            yhat = 1
        if yhat == -1:
            yhat = 0
        return yhat
            
        
