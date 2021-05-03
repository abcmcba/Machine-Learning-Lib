from cs475_types import ClassificationLabel, FeatureVector, Instance, Predictor
import scipy
import numpy
import math

class predictor_adaboost(Predictor):
    def __init__(self, algorithm, num_boosting_iterations):
        self.T = num_boosting_iterations
        self.algorithm = algorithm
        self.alpha = []
        self.weaklearner = []
        
        
    def train(self, instances):
        maxdim = instances[0]._feature_vector.maxindex   # each feature represented by j    
        N = len(instances)
        D = [1.0/N] * N
        trainset = numpy.zeros([N, maxdim])   
        y = []
        c = []
                           
        for i in range(N):
            x = [0] * maxdim
            yy = instances[i]._label.label
            if yy == 0:
                yy = -1
            y.append(yy)
                
            dictpairs = instances[i]._feature_vector.dict.items()
            for keyindex, valuevalue in dictpairs:
                x[keyindex-1] = valuevalue

            trainset[i,:] = x

        for j in range(maxdim):
            c_j = []
            jfeature = trainset[:,j]
            jfeature = numpy.unique(jfeature)
            for k in range(len(jfeature)-1):
                c_j.append(0.5*(jfeature[k] + jfeature[k+1]))
            c.append(c_j)
                

        for t in range(self.T):

            minerror = None
            ccc = 0
            jjj = 0
            yhatleft = {}
            yhatright = {}

            for j in range(maxdim):
                clength = len(c[j])

                # hypothesis class, for any c
                for i in range(clength):
                    cc = c[j][i]    
                    error = 0
                    hy_hat = self.hhat(trainset, y, cc, j, N)
                    h_hat = []

                    # predict yhat
                    for k in range(N):
                        if trainset[k, j] > cc:
                            h_hat.append(hy_hat[0])
                        else:
                            h_hat.append(hy_hat[1])

                    yhatleft[(cc, j)] = hy_hat[0]
                    yhatright[(cc, j)] = hy_hat[1]
                    
                    newerror = 0
                    for n in range(N):
                        if h_hat[n] != y[n]:
                            newerror += D[n] 

                    if newerror < minerror:
                        minerror = newerror
                        ccc = cc
                        jjj = j

            
            self.weaklearner.append([ccc, jjj, yhatleft[(ccc, jjj)], yhatright[(ccc, jjj)]])

            if not minerror:

            '''
            if minerror < 0.000001:
                self.T = t
                break
            '''

                alphat = 0.5 * math.log((1-minerror)/minerror)
                self.alpha.append(alphat)

                Z = 0
                y_hat_seq = []
                for i in range(N):
                    if trainset[i][jjj] > ccc:
                        y_hat_seq.append(yhatleft[(ccc, jjj)])
                    else:
                        y_hat_seq.append(yhatright[(ccc, jjj)])

                    Z += D[i] * math.exp(-alphat*y[i]*y_hat_seq[i])


                for i in range(N):
                    D[i] = D[i] * math.exp(-alphat*y[i]*y_hat_seq[i]) / Z



    def hhat(self, data, paray, parac, paraj, N):
        leftpart = []
        rightpart = []
        hyhat = []
        
        for k in range(N):
            if data[k, paraj] > parac:
                leftpart.append(k)
            else:
                rightpart.append(k)
        
        countnegative = 0
        countpositive = 0

        for i in range(len(leftpart)):
            if paray[leftpart[i]] == 1:
                countpositive += 1
            else:
                countnegative += 1
        
        if countpositive >= countnegative:
            hyhat.append(1)
        else:
            hyhat.append(-1)

        countnegative = 0
        countpositive = 0
        
        for i in range(len(rightpart)):
            if paray[rightpart[i]] == 1:
                countpositive += 1
            else:
                countnegative += 1
        if countpositive >= countnegative:
            hyhat.append(1)
        else:
            hyhat.append(-1)

        return hyhat

         
    
    def predict(self, instance):
        dictpairs = instance._feature_vector.dict.items()
        maxdim = instance._feature_vector.maxindex 
        x = [0] * maxdim
        for keyindex, valuevalue in dictpairs:
            x[keyindex-1] = valuevalue

        targetpositive = 0
        targetnegative = 0

        if self.T == 0:
            ccc_t = self.weaklearner[0][0]
            jjj_t = self.weaklearner[0][1]

            
            if x[jjj_t] > ccc_t:
                yhat = self.weaklearner[0][2]
            else:
                yhat = self.weaklearner[0][3]
            if yhat == -1:
                yhat = 0

        else:
            for t in range(self.T):
                alpha_t = self.alpha[t]
                ccc_t = self.weaklearner[t][0]
                jjj_t = self.weaklearner[t][1]
            
            
                if x[jjj_t] > ccc_t:
                    h_t_x = self.weaklearner[t][2]
                else:
                    h_t_x = self.weaklearner[t][3]

                if h_t_x == 1:
                    targetpositive += alpha_t
                if h_t_x == -1:
                    targetnegative += alpha_t

            if targetpositive >= targetnegative:
                yhat = 1
            else:
                yhat = 0

        return yhat
                

        
        
            
        

