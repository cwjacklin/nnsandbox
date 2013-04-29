nnsandbox -- basic numpy/gpu neural net implementation 
===============================================

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


INSTALLATION
------------
Unzip, cd to the top source directory, and run
    python test_mnist.py
for usage instructions.

The code has only been tested on Windows so far, but the main
only complication for setup on Linux should be building
libcudamat.so inside the gnumpy/ subdirectory.

