from abc import ABCMeta, abstractmethod

# abstract base class for defining labels
class Label:
    __metaclass__ = ABCMeta

    @abstractmethod
    def __str__(self): pass

       
class ClassificationLabel(Label):
    def __init__(self, label):
        # TODO
        self.label = label
        pass
        
    def __str__(self):
        # TODO
        return str(self.label)
        pass

class FeatureVector:
    maxindex = -1
    def __init__(self):
        # TODO
        # construct a dictionary to store data
        self.dict = {}
        pass
        
    def add(self, index, value):
        # TODO
        if self.dict.has_key(index) == False:
            self.dict[index] = value
            FeatureVector.maxindex = max(FeatureVector.maxindex, index)
        else:
            print "Error: more than one value for one feature"
        pass
        
    def get(self, index):
        # TODO
        if self.dict.has_key(index):
            return self.dict[index]
        return 0.0
        

class Instance:
    def __init__(self, feature_vector, label):
        self._feature_vector = feature_vector
        self._label = label

# abstract base class for defining predictors
class Predictor:
    __metaclass__ = ABCMeta

    @abstractmethod
    def train(self, instances): pass

    @abstractmethod
    def predict(self, instance): pass

       
