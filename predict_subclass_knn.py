from cs475_types import ClassificationLabel, FeatureVector, Instance, Predictor
# import scipy
import numpy

class predictor_knn(Predictor):
    def __init__(self, algorithm, knn):
        self.k = knn
        self.algorithm = algorithm
        self.maxdim = None
        self.N = None
        self.testset = []
        self.y = []
        
    def train(self, instances):
        self.maxdim = instances[0]._feature_vector.maxindex    
        self.N = len(instances)

        for i in range(self.N):
            x = [0] * self.maxdim
            self.y.append(instances[i]._label.label)
                                            
            dictpairs = instances[i]._feature_vector.dict.items()
            for keyindex, valuevalue in dictpairs:
                x[keyindex-1] = valuevalue

            self.testset.append(x)



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
        dist = [0] * self.N    
        for i in range(self.N):
            distance = 0.0
            for j in range(self.maxdim):
                distance += (x[j]-self.testset[i][j])**2
            dist[i] = distance ** 0.5

        dist = numpy.array(dist)
        order = numpy.lexsort((self.y,dist))
        rank = order.argsort()

        weight = 1.0 / (1 + dist**2)

        # find the k nearest instances    
        knearest = []
        nearestweight = []
        for idx in range(len(rank)):
            if rank[idx] < self.k:
                knearest.append(self.y[idx])
                nearestweight.append(weight[idx])


        if self.algorithm == "knn":
            count_result = {}
            for i in range(len(knearest)):
                if count_result.has_key(knearest[i]):
                    count_result[knearest[i]] += 1
                else:
                    count_result[knearest[i]] = 1


        elif self.algorithm == "distance_knn":
            count_result = {}
            for i in range(len(knearest)):
                if count_result.has_key(knearest[i]):
                    count_result[knearest[i]] += nearestweight[i]
                else:
                    count_result[knearest[i]] = nearestweight[i]

        result = []
        max_occurence = max(count_result.values())
        for key, value in count_result.items():
            if value == max_occurence:
                result.append(key)

        yhat = min(result)


        return yhat
            
        
