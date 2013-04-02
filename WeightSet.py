from itertools import izip
from operator import mul
from BigMat import *
import numpy as np

##############################################################

class WeightSet(object):
    '''
    A list of all weights defining a NeuralNet.
    Operations:
        += WeightSet
        -= WeightSet
        *= scalar
        len(WeightSet)
    '''
    def __init__(self,cfg):
        if cfg:
            # For each pair of consecutive layers, create a dense set of weights between them
            # Pre-allocate a single block of memory for the weights, so that we can do
            # certain operations like += and *= in one operation, instead of separately for
            # each layer.
            sizes   = [(cfg[k].size+1)*cfg[k+1].size   for k in range(len(cfg)-1) ]
            offsets = [0] + [sum(sizes[:i+1]) for i in range(len(sizes))]
            self._mem = empty((sum(sizes),1))
            self._layers = [DenseWeights(cfg[k],cfg[k+1],self._mem[offsets[k]:offsets[k+1]])     for k in range(len(cfg)-1) ]
        else:
            self._layers = None
            self._mem = None

    def copy(self):
        ws = WeightSet(None)
        ws._layers = [layer.copy() for layer in self._layers]
        return ws

    def ravel(self):
        if self._weights != None:
            return self._weights.ravel()
        return np.hstack([layer.ravel() for layer in self._layers])

    def step_by(self,delta,alpha=1.0):
        if self._mem != None and delta._mem != None:
            # Add delta to all weights in a single step
            iaddmul(self._mem,delta._mem,alpha)
        else:
            for layer,dlayer in zip(self._layers,delta._layers):
                iaddmul(layer.W,dlayer.W,alpha)
                iaddmul(layer.b,dlayer.b,alpha)

    def __getitem__(self,i):
        return self._layers[i]

    def __iter__(self):
        return self._layers.__iter__()

    def __len__(self):
        return len(self._layers)

    def __iadd__(self,other):
        can_accelerate = self._mem != None and ((not isinstance(other,WeightSet)) or other._mem != None)
        if can_accelerate:
            iadd(self._mem,other._mem if isinstance(other,WeightSet) else other)
        else:
            for w,v in izip(self._layers,other._layers):
                w += v
        return self

    def __isub__(self,other):
        can_accelerate = self._mem != None and ((not isinstance(other,WeightSet)) or other._mem != None)
        if can_accelerate:
            isub(self._mem,other._mem if isinstance(other,WeightSet) else other)
        else:
            for w,v in izip(self._layers,other._layers):
                w -= v 
        return self

    def __imul__(self,other):
        can_accelerate = self._mem != None and ((not isinstance(other,WeightSet)) or other._mem != None)
        if can_accelerate:
            imul(self._mem,other._mem if isinstance(other,WeightSet) else other)
        else:
            for w in self._layers:
                w *= other
        return self

##############################################################

class DenseWeights(object):
    '''
    A set of dense weights, going from srclayer to dstlayer.
    init_scale is the scale of the random initial weights, centered about zero.
    Operations:
        += DenseWeights
        -= DenseWeights
        *= scalar
        W,b = DenseWeights   (unpacks into ref to weights 'W' and ref to biases 'b')
    '''
    def __init__(self,inlayer,outlayer,mem=None):
        self.inlayer  = inlayer
        self.outlayer = outlayer

        # Initialize to small random values uniformly centered around 0.0
        n,m = inlayer.size,outlayer.size
        scale = outlayer.init_scale
        scale *= 1./(n+1)**.1
        
        # Make W and b views into the memory
        if mem != None:
            self.W = mem[:n*m].reshape((n,m))
            self.b = mem[n*m:].reshape((1,m))
        else:
            self.W = empty((n,m))
            self.b = empty((1,m))
        fill_randn(self.W); imul(self.W,scale)
        fill_randn(self.b); imul(self.b,scale)

        self._tmp_W = None

    def fprop(self,Hin,Hout):
        W,b = self.W,self.b

        # Compute activation function inputs A
        dot(Hin,W,out=Hout)    # A = dot(H[k-1],W)
        iadd(Hout,b)           # A += b

        # Compute activation function outputs f(A), derivative f'(A) while we're at it
        self.outlayer.f.apply(Hout)  # H[k] = f(A), df[k] = f'(A)

    def bprop(self,Din,Dout,Hout,apply_regularizer=None):
        dot_nt(Din,self.W,out=Dout)
              
        # Add gradient contribution of hidden-unit regularizer, if any
        if apply_regularizer != None:
            apply_regularizer(Dout,Hout)
                
        # Multiply Delta by f'(A) from the corresponding layer
        self.outlayer.f.apply_deriv(Hout)
        imul(Dout,Hout)

    def copy(self):
        cp = DenseWeights(self.inlayer,self.outlayer)
        cp.W[:] = self.W[:]
        cp.b[:] = self.b[:]
        return cp

    def ravel(self):
        return np.hstack([as_numpy(self.W).ravel(),as_numpy(self.b).ravel()])

    def get_tmp_W(self):
        if self._tmp_W == None:
            self._tmp_W = (empty(self.W.shape),empty((1,self.W.shape[1])))
        return self._tmp_W

    def __iter__(self):
        return [self.W,self.b].__iter__() # Used so that "W,b = weights" works, for convenience

    def __len__(self):       return self.W.size + self.b.size
    def __getitem__(self,i): return self.W.ravel()[i] if i < self.W.size else self.b.ravel()[i - self.W.size]
    def __setitem__(self,i,value):
        if i < self.W.size: self.W.ravel()[i] = value
        else:               self.b.ravel()[i - self.W.size] = value

    def __iadd__(self,other):
        iadd(self.W,other.W)
        iadd(self.b,other.b)
        return self

    def __isub__(self,other):
        isub(self.W,other.W)
        isub(self.b,other.b)
        return self

    def __imul__(self,alpha):
        imul(self.W,alpha)
        imul(self.b,alpha)
        return self

