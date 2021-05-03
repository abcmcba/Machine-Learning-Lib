from cs475_types import ClassificationLabel, FeatureVector, Instance, Predictor
# import scipy
import numpy
import math

class predictor_nb_clustering(Predictor):
    def __init__(self, algorithm, clustering_training_iterations, num_clusters):
        self.algorithm = algorithm
        self.I = clustering_training_iterations
        self.K = num_clusters
        self.phi = []
        self.mu = []
        self.sigma = []
        self.N = None
                
    def train(self, instances):
        self.maxdim = instances[0]._feature_vector.maxindex    
        self.N = len(instances)
        trainset = []

        # divide data into K folds
        phi = []
        Kfoldset = [[] for i in range(self.K)]
        for i in range(self.N):
            x = [0] * self.maxdim
            dictpairs = instances[i]._feature_vector.dict.items()
            for keyindex, valuevalue in dictpairs:
                x[keyindex-1] = valuevalue

            trainset.append(x)
            
            foldno = i % self.K
            Kfoldset[foldno].append(x)
            
        for i in range(self.K):
            phi.append((len(Kfoldset[i])+1.0)/(self.N+self.K))

        # initialize mu and sigma
        # in case all values for one feature are 0
        S = 0.01 * numpy.var(numpy.array(trainset),axis=0,ddof=1)
        for i in range(len(S)):
            if S[i] == 0.0:
                S[i] = 0.01
        
        # special case: cluster size==0 or cluster size==1
        mu = []
        sigma = []
        for i in range(self.K):
            mutemp = [0.0] * len(S)
            vartemp = [0.0] * len(S)
            if len(Kfoldset[i]) == 0:
                vartemp = S
            elif len(Kfoldset[i]) == 1:
                mutemp = numpy.mean(numpy.array(Kfoldset[i]),axis=0)
                vartemp = S

            # Case: numpy.var(numpy.array(Kfoldset[i]),axis=0,ddof=1) < S, but needed to consider each dimension
            else:
                mutemp = numpy.mean(numpy.array(Kfoldset[i]),axis=0)
                smallidx = [j for j in range(len(S)) if numpy.var(numpy.array(Kfoldset[i]),axis=0,ddof=1)[j] < S[j]]
                if len(smallidx) == 0 :
                    vartemp = numpy.var(numpy.array(Kfoldset[i]),axis=0,ddof=1)
                else:
                    # select a column
                    for j in smallidx:
                        vartemp[j] = S[j]
                    normalidx = [x for x in range(len(S)) if x not in smallidx]
                    for j in normalidx:
                        vartemp[j] = numpy.var(numpy.array([row[j] for row in Kfoldset[i]]),ddof=1)
                    
                    
            mu.append(mutemp)
            sigma.append(vartemp)
            

        for i in range(self.I):
            # E-step

            trainlabel=[]
            for i in range(self.N):
                x_i = trainset[i]
                posterior = []
                for k in range(self.K):
                    likelihood = sum(-numpy.log((2*math.pi*numpy.array(sigma[k]))**0.5)-(x_i-numpy.array(mu[k]))**2/(2*numpy.array(sigma[k])))
                    posterior.append(numpy.log(phi[k]) + likelihood)
                  
                maxposterior = max(posterior)
                label = [k for k in range(self.K) if posterior[k] == maxposterior]
                trainlabel.append(min(label))
                

            # M-step
            Kfoldset = [[] for k in range(self.K)]

            for n in range(self.N):
                Kfoldset[trainlabel[n]].append(trainset[n])

            phi = []
            for k in range(self.K):
                phi.append((len(Kfoldset[k])+1.0)/(self.N+self.K))

            # compute new mean and new variance
            mu = []
            sigma = []
            for k in range(self.K):
                mutemp = [0.0] * len(S)
                vartemp = [0.0] * len(S)
                if len(Kfoldset[k]) == 0:
                    vartemp = S
                elif len(Kfoldset[k]) == 1:
                    mutemp = numpy.mean(numpy.array(Kfoldset[i]),axis=0)
                    vartemp = S

                # Case: numpy.var(numpy.array(Kfoldset[k]),axis=0,ddof=1) < S, but needed to consider each dimension
                else:
                    smallidx = [j for j in range(len(S)) if numpy.var(numpy.array(Kfoldset[k]),axis=0,ddof=1)[j] < S[j]]
                    if len(smallidx) == 0 :
                        mutemp = numpy.mean(numpy.array(Kfoldset[k]),axis=0)
                        vartemp = numpy.var(numpy.array(Kfoldset[k]),axis=0,ddof=1)
                    else:
                        mutemp = numpy.mean(numpy.array(Kfoldset[k]),axis=0)
                        # select a column
                        for j in smallidx:
                            vartemp[j] = S[j]
                        normalidx = [x for x in range(len(S)) if x not in smallidx]
                        for j in normalidx:
                            vartemp[j] = numpy.var(numpy.array([row[j] for row in Kfoldset[k]]),ddof=1)

                mu.append(mutemp)
                sigma.append(vartemp)

        self.phi = phi
        self.mu = mu
        self.sigma = sigma

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


        posterior=[]
        for k in range(self.K):
            likelihood = sum(-numpy.log((2*math.pi*numpy.array(self.sigma[k]))**0.5)-(x-numpy.array(self.mu[k]))**2/(2*numpy.array(self.sigma[k])))
            posterior.append(numpy.log(self.phi[k]) + likelihood)
                 
        maxposterior = max(posterior)
        label = [k for k in range(self.K) if posterior[k] == maxposterior]
        

        # predict the label
        yhat = min(label)
        
        return yhat
            
        
