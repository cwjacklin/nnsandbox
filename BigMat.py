import numpy as np
import numpy.random
import random
import sys,gc
import math,string

_has_psutil = False
try:
    import psutil
    _has_psutil = True
except ImportError:
    pass


random.seed(9876)
numpy.random.seed(5432)

_gnumpy_loaded = False
try:
    import gnumpy as gp
    cudamat = gp.cmat
    _gnumpy_loaded = True
except ImportError:
    pass

default_dtype = 'float32'
backend = None   # NumpyBackend or GnumpyBackend
backend_name = None 
gradcheck_mode = False
device_prop = None

####################################################

class NumpyBackend(object):

    @staticmethod
    def empty(shape,dtype):    return np.empty(shape,dtype=dtype)

    @staticmethod
    def zeros(shape,dtype):    return np.zeros(shape,dtype=dtype)

    @staticmethod
    def ones(shape,dtype):     return np.ones(shape,dtype=dtype)

    @staticmethod
    def rand(*shape):    return np.array(np.random.rand(*shape),default_dtype)

    @staticmethod
    def randn(*shape):   return np.array(np.random.randn(*shape),default_dtype)

    @staticmethod
    def fill_rand(out):  out[:] = np.random.rand(out.shape[0],out.shape[1])

    @staticmethod
    def fill_randn(out): out[:] = np.random.randn(out.shape[0],out.shape[1])

    @staticmethod
    def array(A,dtype):  return np.array(A,dtype=dtype)

    @staticmethod
    def asarray(A,dtype):return np.asarray(A,dtype=dtype)

    @staticmethod
    def as_numpy(A):     return A

    @staticmethod
    def diff(A,axis,out): return np.diff(A,axis=axis,out=out)

    @staticmethod
    def dot(A,B,out):    return np.dot(A,B,out=out)

    @staticmethod
    def dot_tn(A,B,out): return np.dot(A.T,B,out=out)

    @staticmethod
    def dot_nt(A,B,out): return np.dot(A,B.T,out=out)

    @staticmethod
    def square(A,out):   return np.square(A,out=out)

    @staticmethod
    def logistic(A,out):
        if out == None: out = A.copy()
        else:           out[:] = A[:]
        out *= -1
        np.exp(out,out=out)
        out += 1
        NumpyBackend.reciprocal(out,out=out)
        return out

    @staticmethod
    def tanh(A,out):     return np.tanh(A,out=out)

    @staticmethod
    def sqrt(A,out):     return np.sqrt(A,out=out)

    @staticmethod
    def exp(A,out):      return np.exp(A,out=out)

    @staticmethod
    def log(A,out):      return np.log(A,out=out)

    @staticmethod
    def abs(A,out):      return np.abs(A,out=out)

    @staticmethod
    def sign(A,out):     return np.sign(A,out=out)

    @staticmethod
    def relu(A,out,dout): 
        result = np.maximum(0,A,out=out)
        if dout != None:
            np.sign(out,out=dout)
        return result

    @staticmethod
    def logistic_deriv(A,out): return np.subtract(A,np.square(A),out=out)

    @staticmethod
    def tanh_deriv(A,out): 
        if out == None: 
            out = empty(A.shape(),dtype=A.dtype)
        np.square(A,out=out)
        np.subtract(1,out,out=out)
        return out

    @staticmethod
    def max(A,axis,out): return np.max(A,axis=axis,out=out.ravel() if out != None else None)

    @staticmethod
    def min(A,axis,out): return np.min(A,axis=axis,out=out.ravel() if out != None else None)

    @staticmethod
    def sum(A,axis,out): return np.sum(A,axis=axis,out=out.ravel() if out != None else None)

    @staticmethod
    def mean(A,axis,out):return np.mean(A,axis=axis,out=out.ravel() if out != None else None)

    @staticmethod
    def add(A,B,out):       return np.add(A,B,out=out)

    @staticmethod
    def add_nt(A,B,out):  return np.add(A,B.transpose(),out=out)

    @staticmethod
    def iadd(A,B):          A += B

    @staticmethod
    def iaddmul(A,B,alpha): B *= alpha; A += B

    @staticmethod
    def iassign(A,B):       A[:] = B

    @staticmethod
    def subtract(A,B,out):  return np.subtract(A,B,out=out)

    @staticmethod
    def subtract_nt(A,B,out):  return np.subtract(A,B.transpose(),out=out)

    @staticmethod
    def isub(A,B):          A -= B

    @staticmethod
    def multiply(A,B,out):  return np.multiply(A,B,out=out)

    @staticmethod
    def imul(A,B):          A *= B

    @staticmethod
    def divide(A,B,out):    return np.divide(A,B,out=out)

    @staticmethod
    def idiv(A,B):          A /= B

    @staticmethod
    def reciprocal(A,out):  return np.divide(1.,A,out=out)

    @staticmethod
    def transpose(A,out):
        AT = A.transpose()
        if out != None:
            out[:] = AT
        return AT

    @staticmethod
    def maximum(A,B,out):   return np.maximum(A,B,out=out)

    @staticmethod
    def clip_norm(A,maxnorm,axis,temp_mem):
        if axis != 0:
            raise NotImplementedError("normalization of individual rows not yet implemented")
        # If a temporary memory buffer was supplied, use it instead of allocating a new one
        if temp_mem != None:
            T,t = temp_mem 
        else:
            T,t = np.empty(A.shape,dtype=A.dtype),np.empty((1,A.shape[1]),dtype=A.dtype)
                
        # Compute the square of the norm of weights entering each destination unit (norm along rows)
        np.square(A,out=T)
        np.sum(T,axis=0,out=t.ravel())

        # Normalize each W[:,j] to have norm at most maxnorm
        np.maximum(t,maxnorm**2,out=t)   # make sure anything with norm < maxnorm ends up not being scaled
        np.sqrt(t,out=t)
        reciprocal(t,out=t)
        t *= maxnorm
        np.multiply(A,t,out=A)

    @staticmethod
    def dropout(A,B,rate,outA,outB):
        if outA == None: outA = A
        if outB == None: outB = B
        mask = np.random.binomial(1,rate,A.shape)
        multiply(A,mask,out=outA)
        if B != None:
            multiply(B,mask,out=outB)

    @staticmethod
    def composite(I,channels,backgrounds,out):
        raise Exception("compositing not implemented for numpy backend")

#############################################

class GnumpyBackend(object):

    @staticmethod
    def empty(shape,dtype):    return gp.empty(shape,dtype=dtype)

    @staticmethod
    def zeros(shape,dtype):    return gp.zeros(shape,dtype=dtype)

    @staticmethod
    def ones(shape,dtype):     return gp.ones(shape,dtype=dtype)

    @staticmethod
    def rand(*shape):    return gp.rand(*shape)

    @staticmethod
    def randn(*shape):   return gp.randn(*shape)

    @staticmethod
    def fill_rand(out):  out._base.fill_with_rand()

    @staticmethod
    def fill_randn(out): out._base.fill_with_randn()

    @staticmethod
    def array(A,dtype):        return gp.garray(A,dtype=dtype)

    @staticmethod
    def asarray(A,dtype): 
        if dtype == None:
            dtype = A.dtype
        if isinstance(A,gp.garray):
            return A.astype(dtype)
        return gp.garray(A,dtype=dtype)

    @staticmethod
    def as_numpy(A):     return A.as_numpy_array(default_dtype)

    @staticmethod
    def diff(A,axis,out):
        if axis==0:
            if out == None:
                out = gp.empty((A.shape[0]-1,A.shape[1]),dtype=A.dtype)
            A._base_shaped(1).diff_cols(target=out._base_shaped(1))
            return out
        else:
            if out == None:
                out = gp.empty((A.shape[0],A.shape[1]-1),dtype=A.dtype)
            A._base_shaped(1).diff_rows(target=out._base_shaped(1))
            return out


    @staticmethod
    def dot(A,B,out):
        if out == None:
            out = gp.empty((A.shape[0],B.shape[1]),dtype=A.dtype)
        cudamat.dot(B._base_as_2d(),A._base_as_2d(),target=out._base_as_2d())
        return out

    @staticmethod
    def dot_tn(A,B,out):
        if out == None:
            out = gp.empty((A.shape[1],B.shape[1]),dtype=A.dtype)
        cudamat.dot(B._base_as_2d(),A._base_as_2d().T,target=out._base_as_2d())
        return out

    @staticmethod
    def dot_nt(A,B,out):
        # Using B._base_as_2d().T does not work; cudamat returns dimensionality error
        B._base.mat.is_trans = not B._base.mat.is_trans
        if out == None:
            out = gp.empty((A.shape[1],B.shape[1]),dtype=A.dtype)
        cudamat.dot(B._base_as_2d(),A._base_as_2d(),target=out._base_as_2d())
        B._base.mat.is_trans = not B._base.mat.is_trans
        return out
    
    @staticmethod
    def square(A,out):
        if out == None:
            out = gp.empty(A.shape,dtype=A.dtype)
        cudamat.square(A._base_as_row(),target=out._base_as_row())
        return out

    @staticmethod
    def _unary(func,A,out):
        if out == None:
            out = gp.empty(A.shape,dtype=A.dtype)
        func(A._base_as_row(),target=out._base_as_row())
        return out

    @staticmethod
    def logistic(A,out): return GnumpyBackend._unary(cudamat.sigmoid,A,out)

    @staticmethod
    def tanh(A,out):     return GnumpyBackend._unary(cudamat.tanh,A,out)

    @staticmethod
    def sqrt(A,out):     return GnumpyBackend._unary(cudamat.sqrt,A,out)

    @staticmethod
    def exp(A,out):      return GnumpyBackend._unary(cudamat.exp,A,out)

    @staticmethod
    def log(A,out):      return GnumpyBackend._unary(cudamat.log,A,out)

    @staticmethod
    def abs(A,out):      return GnumpyBackend._unary(cudamat.abs,A,out)

    @staticmethod
    def sign(A,out):     return GnumpyBackend._unary(cudamat.CUDAMatrix.sign,A,out)

    @staticmethod
    def relu(A,out,dout): 
        cudamat.relu(A._base_as_row(),
                     target =( out._base_as_row() if  out != None else None),
                     dtarget=(dout._base_as_row() if dout != None else None))

    @staticmethod
    def logistic_deriv(A,out): return GnumpyBackend._unary(cudamat.sigmoid_deriv,A,out)

    @staticmethod
    def tanh_deriv(A,out): return GnumpyBackend._unary(cudamat.tanh_deriv,A,out)

    @staticmethod
    def max(A,axis,out):
        if A.ndim == 2: 
            if out == None:
                out = gp.empty((A.shape[0],1) if axis == 1 else (1,A.shape[1]),dtype=A.dtype)
            A._base_shaped(1).max(1-axis,target=out._base_shaped(1))
            return out
        else:
            r = gp.max(A,axis)  # gnumpy has optimized max over 1D vectors, so use it
            if out != None:
                assert(out.size == 1)
                out[:] = r[:]
            return r

    @staticmethod
    def min(A,axis,out):
        if A.ndim == 2: 
            if out == None:
                out = gp.empty((A.shape[0],1) if axis == 1 else (1,A.shape[1]),dtype=A.dtype)
            A._base_shaped(1).min(1-axis,target=out._base_shaped(1))
            return out
        else:
            r = gp.min(A,axis)  # gnumpy has optimized max over 1D vectors, so use it
            if out != None:
                assert(out.size == 1)
                out[:] = r[:]
            return r

    @staticmethod
    def sum(A,axis,out):
        if A.ndim == 2: 
            if out == None:
                out = gp.empty((A.shape[0],1) if axis == 1 else (1,A.shape[1]),dtype=A.dtype)
            cudamat.sum(A._base_shaped(1),1-axis,target=out._base_shaped(1))
            return out
        else:
            r = gp.sum(A,axis)  # gnumpy has optimized sum over 1D vectors, so use it
            if out != None:
                assert(out.size == 1)
                out[:] = r[:]
            return r

    @staticmethod
    def mean(A,axis,out):
        out = GnumpyBackend.sum(A,axis,out)
        GnumpyBackend.imul(out,1./A.shape[axis])
        return out

    @staticmethod
    def _add(A,B,out):
        if out == None:
            out = gp.empty(A.shape,dtype=A.dtype)
        if np.isscalar(B): 
            A._base_shaped(1).add(B,target=out._base_shaped(1))
        elif B.shape == A.shape:
            A._base_shaped(1).add(B._base_shaped(1),target=out._base_shaped(1))
        elif (B.ndim == 1 or B.shape[0] == 1) and B.size == A.shape[1]:
            A._base_shaped(1).add_col_vec(B._base_shaped(1),target=out._base_shaped(1))
        elif (B.ndim == 1 or B.shape[1] == 1) and B.size == A.shape[0]:
            A._base_shaped(1).add_row_vec(B._base_shaped(1),target=out._base_shaped(1))
        else:
            raise Exception("unhandled case")
        return out

    @staticmethod
    def add(A,B,out):
        # turn vec + matrix into matrix + vec
        if not np.isscalar(B) and (A.ndim < B.ndim or A.shape[0] < B.shape[0] or A.shape[1] < B.shape[1]):
            A,B = B,A
        return GnumpyBackend._add(A,B,out)

    @staticmethod
    def add_nt(A,B,out):
        if out == None:
            out = gp.empty(A.shape,dtype=A.dtype)
        A._base_shaped(1).add_transpose(B._base_shaped(1),target=out._base_shaped(1))
        return out

    @staticmethod
    def iadd(A,B):          GnumpyBackend._add(A,B,A)

    @staticmethod
    def iaddmul(A,B,alpha): A._base_shaped(1).add_mult(B._base_shaped(1),alpha)

    @staticmethod
    def iassign(A,B):       A._base_shaped(1).assign(B if np.isscalar(B) else B._base_shaped(1))

    @staticmethod
    def subtract(A,B,out):
        if out == None:
            out = gp.empty(A.shape,dtype=A.dtype)
        if np.isscalar(B):
            A._base_shaped(1).subtract(B,target=out._base_shaped(1))
        elif B.shape == A.shape:
            A._base_shaped(1).subtract(B._base_shaped(1),target=out._base_shaped(1))
        elif (B.ndim == 1 or B.shape[0] == 1) and (A.ndim == 1 or B.size == A.shape[1]):
            A._base_shaped(1).subtract_col_vec(B._base_shaped(1),target=out._base_shaped(1))
        elif (B.ndim == 1 or B.shape[1] == 1) and B.size == A.shape[0]:
            A._base_shaped(1).subtract_row_vec(B._base_shaped(1),target=out._base_shaped(1))
        else:
            raise Exception("unhandled case")
        return out

    @staticmethod
    def subtract_nt(A,B,out):
        if out == None:
            out = gp.empty(A.shape,dtype=A.dtype)
        A._base_shaped(1).subtract_transpose(B._base_shaped(1),target=out._base_shaped(1))
        return out

    @staticmethod
    def isub(A,B):          GnumpyBackend.subtract(A,B,A)

    @staticmethod
    def _multiply(A,B,out):
        if out == None:
            out = gp.empty(A.shape,dtype=A.dtype)
        if np.isscalar(B): 
            A._base_shaped(1).mult(B,target=out._base_shaped(1))
        elif B.shape == A.shape:
            A._base_shaped(1).mult(B._base_shaped(1),target=out._base_shaped(1))
        elif (B.ndim == 1 or B.shape[0] == 1) and B.size == A.shape[1]:
            A._base_shaped(1).mult_by_col(B._base_shaped(1),target=out._base_shaped(1))
        elif (B.ndim == 1 or B.shape[1] == 1) and B.size == A.shape[0]:
            A._base_shaped(1).mult_by_row(B._base_shaped(1),target=out._base_shaped(1))
        else:
            raise Exception("unhandled case")
        return out

    @staticmethod
    def multiply(A,B,out):
        # turn vec * matrix into matrix * vec
        if not np.isscalar(B) and (A.ndim < B.ndim or A.shape[0] < B.shape[0] or A.shape[1] < B.shape[1]):
            A,B = B,A
        return GnumpyBackend._multiply(A,B,out)

    @staticmethod
    def imul(A,B):         GnumpyBackend._multiply(A,B,A)

    @staticmethod
    def divide(A,B,out):
        if out == None:
            out = gp.empty(A.shape,dtype=A.dtype)
        if np.isscalar(B):         A._base_shaped(1).divide(B,target=out._base_shaped(1))
        elif A.shape == B.shape:   A._base_shaped(1).divide(B._base_shaped(1),target=out._base_shaped(1))
        else: raise NotImplementedError("broadcasted division not implemented by cudamat")
        return out

    @staticmethod
    def idiv(A,B):          GnumpyBackend.divide(A,B,A)

    @staticmethod
    def reciprocal(A,out):
        if out == None:
            out = gp.empty(A.shape,dtype=A.dtype)
        A._base_as_row().reciprocal(out._base_as_row())
        return out

    @staticmethod
    def transpose(A,out):
        if out == None:
            out = gp.empty((A.shape[1],A.shape[0]),dtype=A.dtype)
        A._base_shaped(1).transpose(out._base_shaped(1))
        return out

    @staticmethod
    def maximum(A,B,out):
        if out == None:
            out = gp.empty(A.shape,dtype=A.dtype)
        if np.isscalar(A) and not np.isscalar(B):
            A,B = B,A
        if np.isscalar(B): A._base_shaped(1).maximum(B,target=out._base_shaped(1))
        else:              A._base_shaped(1).maximum(B._base_shaped(1),target=out._base_shaped(1))
        return out

    @staticmethod
    def clip_norm(A,maxnorm,axis,temp_mem):
        if axis != 0:
            raise NotImplementedError("normalization of individual rows not yet implemented")
        # If a temporary memory buffer was supplied, use it instead of allocating a new one
        if temp_mem != None:
            T,t = temp_mem 
        else:
            T,t = empty(A.shape,dtype=A.dtype),empty((1,A.shape[1]),dtype=A.dtype)
                
        # Compute the square of the norm of weights entering each destination unit (norm along rows)
        square(A,out=T)
        sum(T,axis=0,out=t)

        # Rescale any W[:,j] to have norm at most maxnorm 
        A._base_shaped(1).clip_norm(t._base_shaped(1),maxnorm,target=A._base_shaped(1))

    @staticmethod
    def dropout(A,B,rate,outA,outB):
        if outA == None: outA = A
        if outB == None: outB = B
        if B != None:
            cudamat.dropout(A._base_shaped(1),B._base_shaped(1),rate,
                            targetA=outA._base_shaped(1),
                            targetB=outB._base_shaped(1))
        else:
            cudamat.dropout(A._base_shaped(1),None,rate,
                            targetA=outA._base_shaped(1),
                            targetB=None)

    @staticmethod
    def composite(I,channels,backgrounds,out):
        cudamat.composite(I._base_shaped(1),channels._base_shaped(1),backgrounds._base_shaped(1),
                          out._base_shaped(1))

    @staticmethod
    def cauchy(A,lambd,beta,out):
        if out == None: 
            out = A
        cudamat.cauchy(A._base_as_row(),lambd,beta,out._base_as_row())
        return out



###############################################################
# Provide versions of numpy/gnumpy functions with "out" arguments
# since current version of gnumpy doesn't support 'out' functions
# (even though I hacked a few of them to support it)
# 
#
# These seemingly trivial mulx/addx functions exist because
# using A *= scalar with a gnumpy matrix creates extra copy_kernel instances
# on the GPU and seems slightly slower.
#


def empty(shape,dtype=default_dtype):          return backend.empty(shape,dtype)
def zeros(shape,dtype=default_dtype):          return backend.zeros(shape,dtype)
def ones(shape,dtype=default_dtype):           return backend.ones(shape,dtype)
def rand(*shape):          return backend.rand(*shape)
def randn(*shape):         return backend.randn(*shape)
def fill_rand(out):        return backend.fill_rand(out)
def fill_randn(out):       return backend.fill_randn(out)
def array(A,dtype=None):              return backend.array(A,dtype)         # new copy of A
def asarray(A,dtype=None):            return backend.asarray(A,dtype)       # new *view* of A
def as_numpy(A):           return backend.as_numpy(A)
def diff(A,axis=0,out=None):return backend.diff(A,axis,out)
def dot(A,B,out=None):     return backend.dot(A,B,out)
def dot_tn(A,B,out=None):  return backend.dot_tn(A,B,out)
def dot_nt(A,B,out=None):  return backend.dot_nt(A,B,out)
def square(A,out=None):    return backend.square(A,out)   if not np.isscalar(A) else A*A
def logistic(A,out=None):  return backend.logistic(A,out) if not np.isscalar(A) else 1./(1+np.exp(-A))
def tanh(A,out=None):      return backend.tanh(A,out)     if not np.isscalar(A) else np.tanh(A)
def sqrt(A,out=None):      return backend.sqrt(A,out)     if not np.isscalar(A) else np.sqrt(A)
def exp(A,out=None):       return backend.exp(A,out)      if not np.isscalar(A) else np.exp(A)
def log(A,out=None):       return backend.log(A,out)      if not np.isscalar(A) else np.log(A)
def abs(A,out=None):       return backend.abs(A,out)      if not np.isscalar(A) else np.abs(A)
def sign(A,out=None):      return backend.sign(A,out)     if not np.isscalar(A) else np.sign(A)
def relu(A,out=None,dout=None): return backend.relu(A,out,dout) if not np.isscalar(A) else max(0,A)
def logistic_deriv(A,out=None): return backend.logistic_deriv(A,out) if not np.isscalar(A) else A*(1-A)
def tanh_deriv(A,out=None): return backend.tanh_deriv(A,out) if not np.isscalar(A) else 1-A**2
def max(A,axis=0,out=None):return __builtins__['max'](A) if isinstance(A,list) else backend.max(A,axis,out)
def min(A,axis=0,out=None):return __builtins__['min'](A) if isinstance(A,list) else backend.min(A,axis,out)
def sum(A,axis=0,out=None):return __builtins__['sum'](A) if isinstance(A,list) else backend.sum(A,axis,out)
def mean(A,axis=0,out=None):return backend.mean(A,axis,out)
def add(A,B,out=None):     return backend.add(A,B,out)       # A + B
def add_nt(A,B,out=None):return backend.add_nt(A,B,out)      # A + B.transpose()
def iadd(A,B):             return backend.iadd(A,B)          # A += B
def iaddmul(A,B,alpha):    return backend.iaddmul(A,B,alpha) # A += B*alpha (WARNING: value stored in B is undefined after this)
def iassign(A,B):    return backend.iassign(A,B)
def subtract(A,B,out=None):return backend.subtract(A,B,out)  # A - B
def subtract_nt(A,B,out=None):return backend.subtract_nt(A,B,out)  # A - B.transpose()
def isub(A,B):             return backend.isub(A,B)          # A -= B
def multiply(A,B,out=None):return backend.multiply(A,B,out)  # A * B
def imul(A,B):             return backend.imul(A,B)          # A *= B
def divide(A,B,out=None):  return backend.divide(A,B,out)    # A / B
def idiv(A,B):             return backend.idiv(A,B)          # A /= B
def reciprocal(A,out=None):return backend.reciprocal(A,out)  # 1. / A 
def transpose(A,out=None):return backend.transpose(A,out)
def maximum(A,B,out=None): return backend.maximum(A,B,out)
def clip_norm(A,maxnorm,axis=0,temp_mem=None): return backend.clip_norm(A,maxnorm,axis,temp_mem)   # A[:,i] ./= max(eps,sum(A[:,i]**2))
def dropout(A,B,rate,outA=None,outB=None): return backend.dropout(A,B,rate,outA,outB)
def composite(I,channels,backgrounds,out): return backend.composite(I,channels,backgrounds,out)
def cauchy(A,lambd,beta,out=None): return backend.cauchy(A,lambd,beta,out)

###################################################

def set_backend(name,dtype='float32',device=None):
    global backend
    global backend_name
    global default_dtype
    global device_prop
    global _gnumpy_loaded
    if name == 'gnumpy':
        assert(dtype == 'float32')
        if not _gnumpy_loaded:
            print "warning: cannot set backend to gnumpy; module 'gnumpy' failed to import; using numpy instead"
            return
        backend = GnumpyBackend
        backend_name = name
        default_dtype = 'float32'
        if device == None:
            device = 0
        cudamat.cublas_shutdown()
        cudamat.cuda_set_device(device)
        cudamat.cuda_device_reset()
        cudamat.cublas_init()
        cudamat.CUDAMatrix.init_random(random.randint(0,100))
        device = cudamat.cuda_get_device()
        minfo = memory_info()
        device_prop = cudamat.cuda_get_device_prop(device)
        print "Device %d: %s (%s/%s)" % (device,device_prop.name,
                                         _format_memsize(minfo.gpu_avail,fmt="2.2cM"),
                                         _format_memsize(device_prop.totalGlobalMem,fmt="2.2cM"))

    elif name == 'numpy':
        backend = NumpyBackend
        backend_name = name
        default_dtype = dtype
    else:
        raise ValueError("unrecognized backend '%s'" % name)

def garbage_collect():
    global _gnumpy_loaded
    if _gnumpy_loaded:
        gp.free_reuse_cache(True)
    gc.collect()

class MemoryInfo(object):
    def __init__(self):
        self.cpu_avail = None
        self.cpu_total = None
        self.gpu_avail = None
        self.gpu_total = None

    def __repr__(self):
        str = ''
        if self.cpu_avail != None:
            str += 'cpu = %s/%s; ' % (_format_memsize(self.cpu_avail,fmt="2.2cM"),_format_memsize(self.cpu_total,fmt="2.2cM"))
        if self.gpu_avail != None:
            str += 'gpu = %s/%s'   % (_format_memsize(self.gpu_avail,fmt="2.2cM"),_format_memsize(self.gpu_total,fmt="2.2cM"))
        return str

def memory_info(gc=False):
    global _has_psutil

    if gc: 
        garbage_collect()

    meminfo = MemoryInfo()

    if _has_psutil: 
        vmem = psutil.virtual_memory()
        meminfo.cpu_avail = vmem.available
        meminfo.cpu_total = vmem.total

    if _gnumpy_loaded:
        gmem = cudamat.cuda_memory_info()
        meminfo.gpu_avail = gmem[0]
        meminfo.gpu_total = gmem[1]

    return meminfo


# copied from http://code.activestate.com/recipes/578323-human-readable-filememory-sizes-v2/

def _format_memsize(val,fmt=".2cM"):
    """ define a size class to allow custom formatting
        format specifiers supported : 
            em : formats the size as bits in IEC format i.e. 1024 bits (128 bytes) = 1Kib 
            eM : formats the size as Bytes in IEC format i.e. 1024 bytes = 1KiB
            sm : formats the size as bits in SI format i.e. 1000 bits = 1kb
            sM : formats the size as bytes in SI format i.e. 1000 bytes = 1KB
            cm : format the size as bit in the common format i.e. 1024 bits (128 bytes) = 1Kb
            cM : format the size as bytes in the common format i.e. 1024 bytes = 1KB
    """
    # work out the scale, suffix and base        
    factor, suffix = (8, "b") if fmt[-1] in string.lowercase else (1,"B")
    base = 1024 if fmt[-2] in ["e","c"] else 1000

    # Add the i for the IEC format
    suffix = "i"+ suffix if fmt[-2] == "e" else suffix

    mult = ["","K","M","G","T","P"]

    val = float(val) * factor
    i = 0 if val < 1 else int(math.log(val, base))+1
    v = val / math.pow(base,i)
    v,i = (v,i) if v > 0.5 else (v*base,i-1)

    # Identify if there is a width and extract it
    width = "" if fmt.find(".") == -1 else fmt[:fmt.index(".")]        
    precis = fmt[:-2] if width == "" else fmt[fmt.index("."):-2]

    # do the precision bit first, so width/alignment works with the suffix
    t = ("{0:{1}f}"+mult[i]+suffix).format(v, precis) 

    return "{0:{1}}".format(t,width) if width != "" else t


def set_gradcheck_mode(mode):
    global gradcheck_mode
    gradcheck_mode = mode
    if mode:
        set_backend("numpy","float64")  # need full precision to get sensible numbers out of gradcheck

def get_gradcheck_mode():
    global gradcheck_mode
    return gradcheck_mode

def seed_rand(seed):
    global _gnumpy_loaded
    random.seed(seed)
    numpy.random.seed(seed*7)
    if _gnumpy_loaded:
        gp.seed_rand(seed*13)

def sync_backend(): 
    '''Manually calls cudaSynchronizeThreads() if using cuda, else does nothing'''
    global _gnumpy_loaded
    if _gnumpy_loaded:
        cudamat.cuda_sync_threads()

def reset_backend():
    global _gnumpy_loaded
    if _gnumpy_loaded:
        cudamat.cuda_device_reset()


set_backend('numpy')
seed_rand(9876)



