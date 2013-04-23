from numpy import *
import BigMat as bm
import cPickle as cp
import zlib
from Util import *
from time import time as now

class BatchSet(object):
    def __init__(self,X,Y,S,batchsize):
        self._size = m = X.shape[0]
        self._X = X
        self._Y = Y
        self._S = S
        self._index = 0
        self._Xbuf = None
        self._batchsize = batchsize
        self._blocksize = batchsize * max(1,2048//batchsize)
        self._batches_all = vstack([arange(0        ,m          ,batchsize),
                                    arange(batchsize,m+batchsize,batchsize)]).transpose()
        self._batches_all[-1,-1] = m
        self._batches = self._batches_all.copy()

    def __iter__(self):
        self._index = 0
        return self

    def resize(self,newsize):
        self._batches = self._batches_all[:newsize,:].copy()

    def next(self):
        if self._index >= len(self._batches):
            raise StopIteration
        s = slice(*(self._batches[self._index]))
        Xbuf = self._X[s,:]
        if Xbuf.dtype == 'uint8':
            if self._Xbuf == None:
                self._Xbuf = bm.empty((self._batchsize,self._X.shape[1]))
            self._Xbuf[:s.stop-s.start,:] = Xbuf  # copy
            Xbuf = self._Xbuf[:s.stop-s.start,:]  # point to copy
            bm.imul(Xbuf,1./255)

        batch = DataFold(Xbuf,self._Y[s,:],self._S[s,:] if self._S != None else None)
        self._index += 1
        return batch

    def __len__(self):
        return len(self._batches)

    def shuffle(self):
        random.shuffle(self._batches)

class DataFold(object):
    '''
    A simple structure containing a subset of inputs X,
    and the corresponding target outputs Y
    '''
    def __init__(self,X,Y,S=None):
        assert(X.shape[0] == Y.shape[0])
        self.X = X
        self.Y = Y
        self.S = S
        self.size = X.shape[0]

    def __iter__(self):
        return [self.X,self.Y].__iter__()   # let X,Y = data unpack

    def make_batches(self,batchsize):
        return BatchSet(self.X,self.Y,self.S,min(self.X.shape[0],batchsize))


class DataSet(object):
    '''
    A simple structure containing three DataFold instances:
    a 'train', 'valid', and 'test.
    '''
    def __init__(self,X,Y,Xshape=None,Yshape=None,Xrange=None,Yrange=None,shuffle=True,max_batchsize=1):
        if shuffle:
            perm = random.permutation(X.shape[0])
            X = take(X,perm,axis=0)
            if not (X is Y):
                Y = take(Y,perm,axis=0)
        t0 = now()
        self._X = bm.asarray(X)
        self._Y = bm.asarray(Y) if not (X is Y) else self._X
        print "Host->Device transfer of dataset took %.3fs" % (now()-t0)
        self._size  = X.shape[0]
        self._Xrescale = (1.,0.) #scale,bias
        self.max_batchsize = max_batchsize
        self.Xshape = Xshape or (1,X.shape[1])
        self.Yshape = Yshape or (1,Y.shape[1])
        self.Xdim   = X.shape[1]
        self.Ydim   = Y.shape[1]
        self.Xrange = Xrange or (X.min(axis=0),X.max(axis=0))
        self.Yrange = Yrange or (Y.min(axis=0),Y.max(axis=0))
        if not isscalar(self.Xrange[0]):
            self.Xrange = (bm.asarray(self.Xrange[0]).reshape((1,-1)),bm.asarray(self.Xrange[1]).reshape((1,-1)))
        if not isscalar(self.Yrange[0]):
            self.Yrange = (bm.asarray(self.Yrange[0]).reshape((1,-1)),bm.asarray(self.Yrange[1]).reshape((1,-1)))
        rs = self._rowslice(0,self._size)
        self.train = DataFold(self._X[rs,:],self._Y[rs,:])
        self.valid = DataFold(self._X[0:0,:],self._Y[0:0,:])
        self.test  = DataFold(self._X[0:0,:],self._Y[0:0,:])

    def keys(self):   return ['train','valid','test']
    def values(self): return [self.train,self.valid,self.test]
    def items(self):  return zip(self.keys(),self.values())

    def _rowslice(self,a,b):
        # round endpoint down
        b = a + self.max_batchsize *((b-a) // self.max_batchsize)
        return slice(a,b)

    def __getitem__(self,key):
        if   key == 'train': return self.train
        elif key == 'valid': return self.valid
        elif key == 'test':  return self.test
        raise KeyError("invalid key for DataSet fold")

    def split(self,trainsplit,validsplit=0,testsplit=0):
        assert(trainsplit + validsplit + testsplit <= 100)
        trainsize = int(trainsplit * self._size // 100)
        validsize = int(validsplit * self._size // 100)
        testsize  = int(testsplit  * self._size // 100)
        trs = self._rowslice(0,trainsize)
        vas = self._rowslice(trainsize,trainsize+validsize) 
        tes = self._rowslice(trainsize+validsize,trainsize+validsize+testsize) 
        self.train.X    = self._X[trs,:]
        self.train.Y    = self._Y[trs,:]
        self.train.size = self.train.X.shape[0]
        self.valid.X    = self._X[vas,:]
        self.valid.Y    = self._Y[vas,:]
        self.valid.size = self.valid.X.shape[0]
        self.test.X     = self._X[tes,:]
        self.test.Y     = self._Y[tes,:]
        self.test.size  = self.test.X.shape[0]

    def rescale(self,Xrange,Yrange):
        '''
        Rescales the entire dataset so that all inputs X lie within (Xrange[0],Xrange[1])
        and all targets Y lie within (Yrange[0],Yrange[1]).
        The same scaling factor is applied to all folds.
        '''
        if Xrange != self.Xrange and self._X.dtype != 'uint8':
            Xscale = self.Xrange[1]-self.Xrange[0]
            if isscalar(Xscale):
                Xscale = (Xrange[1]-Xrange[0]) / maximum(1e-5,Xscale)
            else:
                bm.maximum(Xscale,1e-5,out=Xscale)
                bm.reciprocal(Xscale,out=Xscale)
                bm.multiply(Xscale,Xrange[1]-Xrange[0],out=Xscale)

            bm.isub(self._X,self.Xrange[0])
            bm.imul(self._X,Xscale)
            bm.iadd(self._X,Xrange[0])

        if Yrange != self.Yrange and not (self._X is self._Y):
            Yscale = self.Yrange[1]-self.Yrange[0]
            if isscalar(Yscale):
                Yscale = (Yrange[1]-Yrange[0]) / maximum(1e-5,Yscale)
            else:
                bm.maximum(Yscale,1e-5,out=Yscale)
                bm.reciprocal(Yscale,out=Yscale)
                bm.multiply(Yscale,Yrange[1]-Yrange[0],out=Yscale)
            bm.isub(self._Y,self.Yrange[0])
            bm.imul(self._Y,Yscale)
            bm.iadd(self._Y,Yrange[0])

        self.Xrange = Xrange
        self.Yrange = Yrange

################################################

def load_mnist(digits=range(10),split=[50,15,35]):
    X,Y = [],[]
    for d in digits:
        for set in ('train','test'):
            # Load all N instances of digit 'd' as a Nx768 row vector of inputs, 
            # and an Nx10 target vector. 
            Xd,Yd = quickload("data/mnist/mnist_%s_%i.pkl" % (set,d))
            Xd = zlib.decompress(Xd) # decompress byte string 
            Yd = zlib.decompress(Yd) # decompress byte string 
            n = len(Xd)/(28*28)
            Xd = ndarray(shape=(n,28*28),buffer=Xd,dtype='uint8') # convert back to numpy array
            Yd = ndarray(shape=(n,10)   ,buffer=Yd,dtype='uint8') # convert back to numpy array
            X.append(Xd)
            Y.append(asarray(Yd[:,digits],dtype='float32'))  # make the output dimensionality match the number of actual targets, for faster training on subsets of digits

    X = vstack(X)
    Y = vstack(Y)

    data = DataSet(X,Y,Xshape=(28,28,1),Xrange=[0.0,255.0],Yrange=[0.0,1.0],shuffle=True)
    data.split(*split)
    return data
    
