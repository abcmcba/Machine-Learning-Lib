import numpy
import math

class ChainMRFPotentials:
    def __init__(self, data_file):
        with open(data_file) as reader:
            for line in reader:
                if len(line.strip()) == 0:
                    continue

                split_line = line.split(" ")
                try:
                    self._n = int(split_line[0])
                except ValueError:
                    raise ValueError("Unable to convert " + split_line[0] + " to integer.")
                try:
                    self._k = int(split_line[1])
                except ValueError:
                    raise ValueError("Unable to convert " + split_line[1] + " to integer.")
                break

            # create an "(n+1) by (k+1)" list for unary potentials
            self._potentials1 = [[-1.0] * ( self._k + 1) for n in range(self._n + 1)]
            # create a "2n by (k+1) by (k+1)" list for binary potentials
            self._potentials2 = [[[-1.0] * (self._k + 1) for k in range(self._k + 1)] for n in range(2 * self._n)]

            for line in reader:
                if len(line.strip()) == 0:
                    continue

                split_line = line.split(" ")

                if len(split_line) == 3:
                    try:
                        i = int(split_line[0])
                    except ValueError:
                        raise ValueError("Unable to convert " + split_line[0] + " to integer.")
                    try:
                        a = int(split_line[1])
                    except ValueError:
                        raise ValueError("Unable to convert " + split_line[1] + " to integer.")
                    if i < 1 or i > self._n:
                        raise Exception("given n=" + str(self._n) + ", illegal value for i: " + str(i))
                    if a < 1 or a > self._k:
                        raise Exception("given k=" + str(self._k) + ", illegal value for a: " + str(a))
                    if self._potentials1[i][a] >= 0.0:
                        raise Exception("ill-formed energy file: duplicate keys: " + line)
                    self._potentials1[i][a] = float(split_line[2])
                elif len(split_line) == 4:
                    try:
                        i = int(split_line[0])
                    except ValueError:
                        raise ValueError("Unable to convert " + split_line[0] + " to integer.")
                    try:
                        a = int(split_line[1])
                    except ValueError:
                        raise ValueError("Unable to convert " + split_line[1] + " to integer.")
                    try:
                        b = int(split_line[2])
                    except ValueError:
                        raise ValueError("Unable to convert " + split_line[2] + " to integer.")
                    if i < self._n + 1 or i > 2 * self._n - 1:
                        raise Exception("given n=" + str(self._n) + ", illegal value for i: " + str(i))
                    if a < 1 or a > self._k or b < 1 or b > self._k:
                        raise Exception("given k=" + self._k + ", illegal value for a=" + str(a) + " or b=" + str(b))
                    if self._potentials2[i][a][b] >= 0.0:
                        raise Exception("ill-formed energy file: duplicate keys: " + line)
                    self._potentials2[i][a][b] = float(split_line[3])
                else:
                    continue

            # check that all of the needed potentials were provided
            for i in range(1, self._n + 1):
                for a in range(1, self._k + 1):
                    if self._potentials1[i][a] < 0.0:
                        raise Exception("no potential provided for i=" + str(i) + ", a=" + str(a))
            for i in range(self._n + 1, 2 * self._n):
                for a in range(1, self._k + 1):
                    for b in range(1, self._k + 1):
                        if self._potentials2[i][a][b] < 0.0:
                            raise Exception("no potential provided for i=" + str(i) + ", a=" + str(a) + ", b=" + str(b))

    def chain_length(self):
        return self._n

    def num_x_values(self):
        return self._k

    def potential(self, i, a, b = None):
        if b is None:
            if i < 1 or i > self._n:
                raise Exception("given n=" + str(self._n) + ", illegal value for i: " + str(i))
            if a < 1 or a > self._k:
                raise Exception("given k=" + str(self._k) + ", illegal value for a=" + str(a))
            return self._potentials1[i][a]

        if i < self._n + 1 or i > 2 * self._n - 1:
            raise Exception("given n=" + str(self._n) + ", illegal value for i: " + str(i))
        if a < 1 or a > self._k or b < 1 or b > self._k:
            raise Exception("given k=" + str(self._k) + ", illegal value for a=" + str(a) + " or b=" + str(b))
        return self._potentials2[i][a][b]


class SumProduct:
    def __init__(self, p):
        self._potentials = p
        # TODO: EDIT HERE
        # add whatever data structures needed
	self.n = p.chain_length()
	self.k = p.num_x_values()
	self.norm = None
	# ChainMRFPotentials.potential(self, i, a, b)
		

    def marginal_probability(self, x_i):
        # TODO: EDIT HERE
        # should return a python list of type float, with its length=k+1, and the first value 0

        f = {}
	for j in range(self.n):
            temp = []
	    for aa in range(self.k):
                temp.append(self._potentials.potential(j+1, aa+1))
            f[j+1] = temp

        for j in range(self.n+1, 2*self.n):
            temp = []
	    for aa in range(self.k):
                btemp = []
                for bb in range(self.k):
                    btemp.append(self._potentials.potential(j, aa+1, bb+1))
                temp.append(btemp)
            f[j] = temp


        mu = {}
        mu[(1, self.n+1)] = f[1]
        mu[(self.n, 2*self.n-1)] = f[self.n]
		
	# from left to right
        for j in range(1, self.n):
            if j < self.n-1:
                mutemp = numpy.array(f[self.n+j]).transpose()*mu[(j,self.n+j)]
                mu[(self.n+j,j+1)] = numpy.sum(mutemp, axis=1)

                mu[(j+1, self.n+j+1)] = numpy.multiply(numpy.array(f[j+1]), numpy.array(mu[(self.n+j,j+1)]))
            else:
                mutemptemp = numpy.array(f[2*self.n-1]).transpose()*mu[(self.n-1,2*self.n-1)]
                mu[(2*self.n-1, self.n)] = numpy.sum(mutemptemp, axis=1)
                
        # from right to left         
        for j in range(self.n-1, 1, -1):
            mutemp = numpy.array(f[self.n+j])*mu[(j+1,self.n+j)]
            mu[(self.n+j,j)] = numpy.sum(mutemp, axis=1)

            mu[(j, self.n+j-1)] = numpy.multiply(numpy.array(f[j]), numpy.array(mu[(self.n+j,j)]))

        mutemptemp = numpy.array(f[self.n+1])*mu[(2,self.n+1)]
        mu[(self.n+1, 1)] = numpy.sum(mutemptemp, axis=1)

        pr = {}

        for i in range(1,self.n+1):
            if i == 1:
                pr[1] = [0] + (numpy.multiply(numpy.array(f[1]), numpy.array(mu[(1+self.n,1)]))).tolist()
            elif i == self.n:
                pr[self.n] = [0] + (numpy.multiply(numpy.array(f[self.n]), numpy.array(mu[(2*self.n-1,self.n)]))).tolist()
                self.norm = sum(pr[self.n])
            else:
                temp = (numpy.multiply(numpy.array(f[i]), numpy.array(mu[(i+self.n,i)]))).tolist()
                pr[i] = [0] + (numpy.multiply(temp, numpy.array(mu[(self.n+i-1,i)]))).tolist()
            pr[i] = numpy.array(pr[i])/sum(pr[i])

        return pr[x_i]        
	
        # This code is used for testing only and should be removed in your implementation.
        # It creates a uniform distribution, leaving the first position 0
		
        # result = [1.0 / (self._potentials.num_x_values())] * (self._potentials.num_x_values() + 1)
        # result[0] = 0
        # return result


class MaxSum:
    def __init__(self, p):
        self._potentials = p
        self._assignments = [0] * (p.chain_length() + 1)
        self.n = p.chain_length()
	self.k = p.num_x_values() 
        # TODO: EDIT HERE
        # add whatever data structures needed

    def get_assignments(self):
        return self._assignments

    def max_probability(self, x_i):
        # TODO: EDIT HERE
	
        ff = {}
	for j in range(self.n):
            temp = []
	    for aa in range(self.k):
                temp.append(math.log(self._potentials.potential(j+1, aa+1)))
            ff[j+1] = temp

        for j in range(self.n+1, 2*self.n):
            temp = []
	    for aa in range(self.k):
                btemp = []
                for bb in range(self.k):
                    btemp.append(math.log(self._potentials.potential(j, aa+1, bb+1)))
                temp.append(btemp)
            ff[j] = temp


        mumu = {}
        mumu[(1, self.n+1)] = ff[1]
        value = {}

        for j in range(2,self.n+1):
            if j < self.n:
                mumu[(self.n+j-1,j)] = (numpy.array(ff[self.n+j-1]).transpose() + mumu[(j-1,self.n+j-1)]).max(axis=1)      
                mumu[(j,self.n+j)] = numpy.array(ff[j]) + numpy.array(mumu[(self.n+j-1,j)])
                temp = numpy.argmax(numpy.array(ff[self.n+j-1]).transpose() + mumu[(j-1,self.n+j-1)], axis=1)+1
                value[j-1] = numpy.concatenate([[0],temp])
            else:
                mumu[(2*self.n-1,self.n)] = (numpy.transpose(numpy.array(ff[2*self.n-1])) + mumu[(self.n-1,2*self.n-1)]).max(axis=1)
                temp = numpy.argmax(numpy.transpose(numpy.array(ff[2*self.n-1])) + mumu[(self.n-1,2*self.n-1)], axis=1)+1
                value[self.n-1] = numpy.concatenate([[0],temp])

        
        value[self.n] = numpy.argmax(numpy.array(ff[self.n]) + mumu[(2*self.n-1,self.n)]) + 1
        self._assignments[self.n] = value[self.n]
        prmax = max(numpy.array(ff[self.n]) + mumu[(2*self.n-1,self.n)])                        
        
        for j in range(self.n-1,0,-1):
            self._assignments[j] = value[j][self._assignments[j+1]]

        sum_product=SumProduct(self._potentials)
        sum_product.marginal_probability(self.n)
        constant = numpy.log(sum_product.norm)
        
        return prmax - constant
