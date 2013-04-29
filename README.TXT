nnsandbox -- basic numpy/gpu neural net implementation 
======================================================

REQUIREMENTS
------------
Python 2.7+ (tested on 64-bit only)
   http://www.python.org/getit/

Numpy 1.7+ (preferably linked with MKL)
   http://www.lfd.uci.edu/~gohlke/pythonlibs/#numpy  (Win64)

SciPy:
   http://www.lfd.uci.edu/~gohlke/pythonlibs/#scipy  (Win64)

Matplotlib:
   http://matplotlib.org/downloads.html

CUDA Runtime and/or SDK:
   https://developer.nvidia.com/cuda-downloads

The code comes with a modified version of Cudamat and Gnumpy 
libraries in the gnumpy/ subdirectory. 
These libraries are needed for the GPU code path to work.
The original, unmodified libraries are available at:

Cudamat: by Vladimir Mnih.
   https://code.google.com/p/cudamat/

Gnumpy: by Tijmen Tieleman.
   http://www.cs.toronto.edu/~tijmen/gnumpy.html


INSTALLATION -- WINDOWS
-----------------------
The code is normally tested on Visual Studio 2010, and so it
comes with libcudamat.dll compiled for x64. To recompile
the DLL, use gnumpy/gnumpy.sln. You must have the CUDA runtime
DLLs (cudart64_50_35.dll,cublas64_50_35.dll,etc...)
somewhere in your path, or libcudamat.dll will fail to load.
For example, with the SDK they are in:
   C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v5.0\bin

Unzip, cd to the top source directory, and run
   python test_mnist.py
for usage instructions.

INSTALLATION -- LINUX
---------------------
Unzip, cd to the top source directory.
You must build gnumpy/libcudamat.so yourself, so you'll need
the full CUDA SDK installed. Steps below provided by Alireza Makhzani.

Make sure you can run "nvcc" from your terminal.
If not, run
   export PATH=$PATH:/usr/local/cuda/bin

Make sure gcc can find your CUDA SDK installation...
For 32-bit installations, you can run
   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib
For 64-bit, make sure the lib64 path comes first:
   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/lib

Go to the gnumpy/ subdirectory, and run "make".
If the above step fails, try to install nose.

Now from the top source directory (parent of gnumpy/) run
   python test_mnist.py
for usage instructions.
