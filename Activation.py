from BigMat import *
from numpy import inf
from Util import TempMatrix

class Activation(object):
    '''
    An activation function.
    An instance f can be evaluated element-wise on input matrix A three ways:
       H = f(A)             # transform A by f(.)
       f(A,out=H)           # same, but matrix H pre-allocated
       f(A,out=H,dout=df)   # compute both H=f(A), and df=f'(A)
    '''
    def ideal_loss(self): return 'mse'

class ActivationLinear(Activation):
    '''Activation function identity(A)'''
    def name(self):         return "linear"
    def ideal_domain(self): return [-1,1]
    def ideal_range(self):  return [-1,1]
    def actual_range(self): return [-inf,inf]

    def __call__(self,A,out,dout):
        if not (out is A):
            out[:] = A[:]
        if dout != None:
            iassign(dout,1)


class ActivationLogistic(Activation):
    '''Activation function sigmoid(A), i.e. logisitic function'''
    def name(self):         return "logistic"
    def ideal_domain(self): return [ 0.0,1.0]
    def ideal_range(self):  return [ 0.1,0.9]
    def actual_range(self): return [ 0.0,1.0]

    def __call__(self,A,out,dout):
        logistic(A,out=out)
        if dout != None:
            logistic_deriv(out,out=dout)


class ActivationTanh(Activation):
    '''Activation function tanh(A)'''
    def name(self):         return "tanh"
    def ideal_domain(self): return [-1.2,1.2]
    def ideal_range(self):  return [-0.9,0.9]
    def actual_range(self): return [-1.0,1.0]

    def __call__(self,A,out,dout):
        tanh(A,out=out)
        if dout != None:
            tanh_deriv(out,out=dout)


class ActivationRelu(Activation):
    '''Activation function max(0,A), i.e. rectified linear'''
    def name(self):         return "relu"
    def ideal_domain(self): return [-1.2,1.2]
    def ideal_range(self):  return [ 0.0,1.0]
    def actual_range(self): return [ 0.0,inf]

    def __call__(self,A,out,dout):
        relu(A,out=out,dout=dout)


class ActivationSoftmax(Activation):
    '''Activation function softmax(A)'''
    def __init__(self):
        self._tmp_denom = TempMatrix()

    def name(self):         return "softmax"
    def ideal_domain(self): return [0.0,1.0]
    def ideal_range(self):  return [0.0,1.0]
    def actual_range(self): return [0.0,1.0]
    def ideal_loss(self):   return 'nll'

    def __call__(self,A,out,dout):
        # First pre-allocate enough memory to accumulate denominator of each sample
        maxval = denom = self._tmp_denom.get_capacity(A.shape[0],1)

        # Then compute logsum softmax (subtract off maximum value)
        max(A,axis=1,out=maxval)
        subtract(A,maxval,out=out)
        exp(out,out=out)
        sum(out,axis=1,out=denom)
        reciprocal(denom,out=denom)
        multiply(out,denom,out=out)


##########################################################

def make_activation(typename):
    if   typename == "linear":   return ActivationLinear()
    elif typename == "logistic": return ActivationLogistic()
    elif typename == "tanh":     return ActivationTanh()
    elif typename == "relu":     return ActivationRelu()
    elif typename == "softmax":  return ActivationSoftmax()
    elif typename == None:       return None
    raise ValueError("unrecognized activation function '%s'" % typename)
