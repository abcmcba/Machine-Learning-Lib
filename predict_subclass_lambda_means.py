from cs475_types import ClassificationLabel, FeatureVector, Instance, Predictor
# import scipy
import numpy

class predictor_lambda_means(Predictor):
    def __init__(self, algorithm, clustering_training_iterations, cluster_lambda):
        self.algorithm = algorithm
        self.I = clustering_training_iterations
        self.clambda = cluster_lambda
        self.mu = []
        self.K = None
        self.maxdim = None

                
    def train(self, instances):
        mutemp = []
        trainset = []
        self.maxdim = instances[0]._feature_vector.maxindex    
        self.N = len(instances)

        for i in range(self.N):
            x = [0] * self.maxdim
                                            
            dictpairs = instances[i]._feature_vector.dict.items()
            for keyindex, valuevalue in dictpairs:
                x[keyindex-1] = valuevalue

            trainset.append(x)

        mutemp.append(numpy.mean(numpy.array(trainset),axis=0))
        
        if self.clambda == 0.0:
            self.clambda = sum(sum((numpy.array(trainset)-numpy.array(mutemp)[0,:])**2)) / float(self.N)

        self.K = 1
        for i in range(self.I):
            # E-step
            r = []
            for j in range(self.N):
                disttemp = []
                xj = trainset[j]
                
                for k in range(self.K):
                    disttemp.append(sum((numpy.array(xj)-numpy.array(mutemp)[k,:])**2))

                rtemp = [0]*self.K
                
                if min(disttemp) <= self.clambda:
                    clusterk = disttemp.index(min(disttemp))
                    rtemp[clusterk] = 1
                    r.append(rtemp)
                else:
                    mutemp.append(xj)
                    self.K += 1
                    clusterk = self.K
                    rtemp.append(1)
                    r.append(rtemp)
            
            # M-step
            mutemp = [[0]*self.maxdim for i in range(self.K)]
            for k in range(self.K):
                temp = []
                
                for j in range(self.N):
                    try:
                        if r[j][k] == 1:
                            temp.append(numpy.array(trainset)[j,:])
                            # mutemp[k] = numpy.mean(numpy.array(temp), axis=0)
                    except IndexError:
                        pass
                    if len(temp) != 0:
                        mutemp[k] = numpy.mean(numpy.array(temp), axis=0)

        self.mu = mutemp
        
                                  
    def predict(self, instance):
        # compute the max index of a test instance
        test_maxdim = -1
        for keyindex in instance._feature_vector.dict.keys():
            test_maxdim = max(test_maxdim, keyindex)

        # initialize test x
        x = [0] * test_maxdim
        
        dictpairs = instance._feature_vector.dict.items()
        for keyindex, valuevalue in dictpairs:
            x[keyindex-1] = valuevalue
            
        # in case dimension is different
        if test_maxdim == self.maxdim:                               
            pass
        elif test_maxdim < self.maxdim:
            x += [0] * (self.maxdim-test_maxdim)
        elif test_maxdim > self.maxdim:
            x = x[:self.maxdim]

        # Euclidean distance
        disttemp = []
        for k in range(self.K):
            disttemp.append(sum((x-numpy.array(self.mu)[k,:])**2))
    
        clusterk = disttemp.index(min(disttemp))
        yhat = clusterk

        return yhat
            
        
