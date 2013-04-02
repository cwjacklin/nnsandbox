from itertools import izip
from operator import mul
from Activation import make_activation
from BigMat import *
from Model import *
import numpy as np

##############################################################

class NeuralNetCfg(object):
    '''
    A list of layer-specifiers, where each layer specifies 
    the type, size, and behaviour of a single network layer.
    Activation is a string: 'logistic','tanh','relu' or 'softmax'.
    Maxnorm is the maximum norm of the weights *entering* 
    each hidden unit of the layer, or None if no limit.
    '''
    def __init__(self,L1=0.0,L2=0.0,maxnorm=None,dropout=None,sparsity=None,init_scale=0.05):
        # These values (L1,L2,maxnorm,dropout) are the defaults for
        # all layers of this neural net, but can be overridden
        # for a specific layer via add_layer
        self.L1 = L1
        self.L2 = L2
        self.maxnorm = maxnorm
        self.dropout = dropout
        self.sparsity = sparsity
        self.init_scale = init_scale
        self.loss = None
        
        self._layers = [None,None]

    def input(self,size,dropout=None):
        layer = NeuralNetLayerCfg(size,None)
        layer.dropout = dropout
        self._layers[0] = layer

    def hidden(self,size,activation,L1=None,L2=None,maxnorm=None,dropout=None,sparsity=None,init_scale=None):
        layer = NeuralNetLayerCfg(size,activation)
        layer.L1 = L1
        layer.L2 = L2
        layer.maxnorm = maxnorm
        layer.dropout = dropout
        layer.sparsity = sparsity
        layer.init_scale = init_scale
        self._layers.insert(-1,layer)

    def output(self,size,activation,L1=None,L2=None,maxnorm=None,init_scale=None,loss=None):
        layer = NeuralNetLayerCfg(size,activation)
        layer.L1 = L1
        layer.L2 = L2
        layer.maxnorm = maxnorm
        layer.dropout = None
        layer.sparsity = None
        layer.init_scale = init_scale
        self._layers[-1] = layer
        self.loss = loss

    def __len__(self):       return len(self._layers)
    def __iter__(self):      return self._layers.__iter__()
    def __getitem__(self,i): return self._layers[i]

    def finalize(self):
        assert(self._layers[0]  != None) # input layer must be specified
        assert(self._layers[-1] != None) # output layer must be specified
        K = len(self._layers)
        for k in range(K):
            layer = self._layers[k]
            layer.f = make_activation(layer.activation)
            if k > 0:
                if layer.L1 == None:       layer.L1 = self.L1
                if layer.L2 == None:       layer.L2 = self.L2
                if layer.maxnorm == None:  layer.maxnorm = self.maxnorm
                if layer.sparsity == None: layer.sparsity = self.sparsity
                if layer.init_scale == None: layer.init_scale = self.init_scale
                if k < K:
                    if layer.dropout == None:  layer.dropout = self.dropout
        return self

    def __repr__(self):  # Useful for printing 
        str = ''
        for name,value in self.__dict__.iteritems():
            if name == '_layers':
                continue
            if (isinstance(value,NeuralNetLayerCfg)):
                str += '{0}=...\n'.format(name)
            elif (isinstance(value,basestring)):
                str += '{0}=\'{1}\'\n'.format(name,value)
            else:
                str += '{0}={1}\n'.format(name,value)
        for k in range(len(self._layers)):
            str += ('layer[%d] = {\n' % k) + self._layers[k].__repr__() + '}\n'
        return str

##############################################################

class NeuralNetLayerCfg(object):
    def __init__(self,size,activation):
        self.shape      = (size,) if isinstance(size,(long,int)) else size  # Shape can be a tuple, like (28,28) for MNIST, for example.
        self.size       = reduce(mul,list(self.shape))         # Size should always be the raw number of units in the layer.
        self.activation = activation
        self.f          = None # will be set to make_activation
        self.L1         = None
        self.L2         = None
        self.maxnorm    = None
        self.dropout    = None
        self.sparsity   = None
        self.init_scale = None

    def __repr__(self):  # Useful for printing 
        str = ''
        for name,value in self.__dict__.iteritems():
            if (name in ('shape','f')) or value == None:
                continue
            if (isinstance(value,basestring)):
                str += '  {0}=\'{1}\'\n'.format(name,value)
            else:
                str += '  {0}={1}\n'.format(name,value)
        return str


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
        return [self.W,self.b].__iter__() # Used so that "W,b = weights" works, for convenience ... unpack W, then b

    def __len__(self):       return self.W.size + self.b.size
    def __getitem__(self,i): return self.W.flat[i] if i < self.W.size else self.b.flat[i - self.W.size]
    def __setitem__(self,i,value):
        if i < self.W.size: self.W.flat[i] = value
        else:               self.b.flat[i - self.W.size] = value

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

##############################################################

class WeightSet(object):
    '''
    A set of all weights defining a NeuralNet.
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

class NeuralNet(Model):
    '''
    A NeuralNet is defined by a sequence of layers.
    Each layer has its own size, and its own activation function.
    Each pair of consecutive layers has its own weights,
    and the set of all weights is contained in the member
    function 'weights'.
    '''
    def __init__(self,cfg):
        self._cfg = cfg.finalize()
        Model.__init__(self,cfg.loss or cfg[-1].f.ideal_loss())

        self.weights = WeightSet(cfg)

        # Each _tmp_H[k] and _tmp_df[k] contains pre-allocated buffers for storing
        # information during forwardprop that is useful during backprop.
        # It is intended to be used on minibatches only, since the
        # gradient (backprop) is never needed on the full training set,
        self._H = None
        self._df = None
        self._tmp_H  = [TempMatrix(1,w.outlayer.size) for w in self.weights]  # H  = f(A)
        self._tmp_df = [TempMatrix(1,w.outlayer.size) for w in self.weights]  # df = f'(A)
        self._tmp_D  = [TempMatrix(1,w.outlayer.size) for w in self.weights]  # Delta for backprop
        self._tmp_R  = [TempMatrix(1,w.outlayer.size) for w in self.weights]  # temp for regularizer
        if self._cfg[0].dropout:
            self._tmp_X  = TempMatrix(1,self._cfg[0].size)  # for dropout on input

    def numlayers(self):
        '''The number of layers, including output but excluding input.'''
        return len(self.weights) # everything but the input layer is a 'layer'

    def ideal_domain(self): return self._cfg[ 1].f.ideal_domain()   # first hidden layer's ideal domain
    def ideal_range(self):  return self._cfg[-1].f.ideal_range()    # output layer's ideal range

    def make_weights(self): 
        ws = WeightSet(self._cfg)
        for layer in ws:
            layer.W *= 0
            layer.b *= 0
        return ws

    def __call__(self,X):
        return self.eval(X)

    def eval(self,X,want_hidden=False):
        '''
        Given (m x n_0) matrix X, evaluate all m inputs on the neural network.
        The result is an (m x n_K) matrix of final outputs, or, if want_hidden is True,
        then a list of all hidden outputs are provided, where the last entry is the matrix of
        final outputs.
        '''
        H = self._fprop(X) 
        if want_hidden:
            return H
        return H[-1] # Return only the final output, unless all layers were explicitly requested

    def grad(self,data,out=None):
        '''
        Compute the gradient of the current loss function (MSE,NLL) with respect to all weights.
        '''
        
        if not self._has_dropout():
            # No dropout, so do a single forward propagation pass.
            # _fprop will store all values needed for a subsequent _bprop call
            self._fprop(data.X,want_grad=True)
            
            # Backpropagate the error signal, producing a gradient for weight matrix.
            out = self._bprop(data.Y,out=out)
        else:
            # _fprop with dropout, and bprop the corresponding gradient contribution of the dropped-out architecture.
            self._fprop(data.X,want_grad=True,dropout_mode="train")
            out = self._bprop(data.Y,out=out,want_loss=True,want_penalty=True,want_reg=False)

            if self._has_regularizer():
                # If we have a regularizer on the hidden activations, it should act on the
                # non-dropped-out values, which means we need to compute a 'normal' _fprop
                # pass and only _bprop the regularizer's contribution
                self._fprop(data.X,want_grad=True,dropout_mode="test")
                out = self._bprop(data.Y,out=out,want_loss=False,want_penalty=False,want_reg=True)
        return out

    def _fprop(self,X,want_grad=False,dropout_mode="test"):
        '''
        Given (m x n_0) matrix X, evaluate all m inputs on the neural network.
        Returns  an (m x n_K) matrix of final outputs.
        '''
        m,n = X.shape; assert(m >= 1); assert(n == self._cfg[0].size)

        # H[k] is an (m x n_k) matrix, where n_k is the number of hidden units in layer k
        H  = [X]    +  self._get_tmp(self._tmp_H,m)
        df = [None] + (self._get_tmp(self._tmp_df,m) if want_grad else [None for layer in self._cfg[1:]])

        # Forward pass, starting from earliest layer, storing all intermediate computations
        for k in range(1,self.numlayers()+1):
            j = k-1            # Hj (prev layer) has well-defined value coming into this loop
            if self._has_dropout():
               H[j] = self._apply_dropout(H[j],df[j],j,dropout_mode)

            W,b = self.weights[k-1]
            A = H[k]           # Hk (this layer) has undefined value; we compute this matrix

            # Compute activation function inputs A
            dot(H[j],W,out=A)  # A = dot(H[j],W)
            iadd(A,b)          # A += b

            # Compute activation function outputs f(A), derivative f'(A) while we're at it
            f = self.weights[k-1].outlayer.f
            f(A,out=H[k],dout=df[k]) # H[k] = f(A), df[k] = f'(A)

        self._H = H
        self._df = df
        return H[1:]  # first element is just X, so discard it

    def _bprop(self,Y,out=None,
               want_loss=True,want_penalty=True,want_reg=True):
        '''
        Compute the gradient of a cost function.
        The "out" argument should be an instance of WeightSet; _bprop will fill
        each layer of 'out' with the gradient of the cost function w.r.t. that 
        layer's current weight matrix.
        ASSUMPTION: _bprop can only be called immediately after _fprop, since _bprop 
                    re-uses values that have been stored during _fprop.
        '''
        dweights = out or self.make_weights()

        # H[k] and df[k] are assumed to have been previously computed in a call to _fprop()
        H  = self._H
        df = self._df
        D  = [None]+ self._get_tmp(self._tmp_D,Y.shape[0])  # D[k] is temporary storage for delta
        R  = [None]+(self._get_tmp(self._tmp_R,Y.shape[0]) if self._has_regularizer() else [None  for layer in self._cfg[1:]])

        # Calculate initial Delta based on loss function, outputs Z=H[-1] and targets Y
        if want_loss:
            self._loss_delta(H[-1],Y,df[-1],out=D[-1])
        else:
            D[-1] *= 0

        # Backward pass
        for k in reversed(range(1,self.numlayers()+1)):
            j = k-1
            Wk,bk = self.weights[k-1]
            dW,db = dweights[k-1]

            # Compute gradient contribution of loss function
            if want_loss:
                dot_tn(H[j],D[k],out=dW)
                sum(D[k],axis=0,out=db)
            else:
                tmp_dW,tmp_db = self.weights[k-1].get_tmp_W()
                dot_tn(H[j],D[k],out=tmp_dW)
                sum(D[k],axis=0,out=tmp_db)
                iadd(dW,tmp_dW)
                iadd(db,tmp_db)

            # Add gradient contribution of penalty 
            if want_penalty:
                self._penalty_grad(self.weights[k-1],dW)

            # Compute the Delta value for the next iteration k-1
            if k > 1:
                dot_nt(D[k],Wk,out=D[j])
              
                # Add gradient contribution of hidden-unit regularizer
                if want_reg:
                    self._regularizer_delta(j,H[j],D[j],R[j])
                
                # Multiply Delta by f'(A) from the corresponding layer
                imul(D[j],df[j])

        return dweights

    def _apply_dropout(self,H,df,k,mode):
        dropout_rate = self._cfg[k].dropout
        if dropout_rate:
            Hsrc = H
            if k == 0: # If dropout on input, need temp storage so that we don't destroy the input
                H = self._get_tmp(self._tmp_X,Hsrc.shape[0])
            if mode == "train":
                dropout(Hsrc,df,dropout_rate,outA=H,outB=df)
            else:
                multiply(Hsrc,(1-dropout_rate),out=H)
                if df != None:
                    imul(df,(1-dropout_rate))
        return H


    def apply_constraints(self):
        self._constrain_weights()

    def _has_dropout(self):
        return [ bool(layer.dropout)  for layer in self._cfg ].count(True) > 0


    ############### REGULARIZER ###############

    def regularizer(self,H):
        '''Returns the sum of all regularization costs on hidden units (sparsity cost)'''
        if not self._has_regularizer():
            return 0.0
        R = self._get_tmp(self._tmp_R,H[0].shape[0])
        cost = 0.0
        for k in range(self.numlayers()):
            lambd,alpha = self._cfg[k+1].sparsity or (0.0,1.0)
            if lambd == 0.0:
                continue
            square(H[k],out=R[k])
            add(R[k],alpha**2,out=R[k])
            log(R[k],out=R[k])           # log(h^2 + alpha^2)
            cost += lambd*as_numpy(sum(R[k].ravel()))/H[k].shape[0]
            cost -= lambd*H[k].shape[1]*log(alpha**2)  # subtract off a constant to make perfect sparsity have zero cost
        return cost

    def _regularizer_delta(self,k,Hk,Dk,Rk):
        '''
        Adds the hidden-unit regularizer contribution to the delta matrix Dk
        for layer k, based on hidden activations Hk
        '''
        lambd,alpha = self._cfg[k].sparsity or (0.0,1.0)
        if lambd == 0.0:
            return
        square(Hk,out=Rk)
        iadd(Rk,alpha**2)
        divide(Hk,Rk,out=Rk)
        iaddmul(Dk,Rk,lambd * 2. / Hk.shape[0])   # D += lambda * 2/m * H ./ (H.^2 + alpha^2)

    def _has_regularizer(self):
        return [bool(layer.sparsity) for layer in self._cfg].count(True) > 0

    ############### PENALTY ###############

    def penalty(self):
        if not self._has_penalty():
            return 0.0
        L1 = L2 = 0.0
        for layer in self.weights:
            if layer.outlayer.L1 > 0.0:
                absW,_ = layer.get_tmp_W()
                abs(layer.W,out=absW)
                L1 += layer.outlayer.L1*as_numpy(sum(absW.ravel()))     # L1 * sum(abs(W))
            if layer.outlayer.L2 > 0.0:
                sqrW,_ = layer.get_tmp_W()
                square(layer.W,out=sqrW)
                L2 += layer.outlayer.L2*0.5*as_numpy(sum(sqrW.ravel())) # L2 * 0.5 * sum(W.^2)
        return L1 + L2

    def _penalty_grad(self,layer,dW):
        if layer.outlayer.L1 > 0.0:
            W,_ = layer.get_tmp_W()
            sign(layer.W,out=W)
            iaddmul(dW,W,layer.outlayer.L1)      # dW += L1 * sign(W)
        if layer.outlayer.L2 > 0.0:
            W,_ = layer.get_tmp_W()
            multiply(layer.W,layer.outlayer.L2,out=W)
            iadd(dW,W)                           # dW += L2 * W

    def _has_penalty(self):
        return [bool(layer.L1) or bool(layer.L2) for layer in self._cfg].count(True) > 0

    ###################### UTILITY FUNCTIONS ######################

    def _constrain_weights(self):
        for weights in self.weights:
            if weights.outlayer.maxnorm:
                # Normalize each column in the weight matrix, only if its magnitude is > maxnorm
                clip_norm(weights.W,axis=0,maxnorm=weights.outlayer.maxnorm,temp_mem=list(weights.get_tmp_W()))


    def _get_tmp(self,temps,m=-1):
        if m == -1:
            return [ temp.get() for temp in temps ] if isinstance(temps,list) else temps.get()
        else:
            return [ temp.get_capacity(m) for temp in temps ] if isinstance(temps,list) else temps.get_capacity(m)

