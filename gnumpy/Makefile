cudamat:
	nvcc -O -Xlinker -rpath,/usr/local/cuda/lib --ptxas-options=-v --compiler-options '-fPIC' -o libcudamat.so --shared cudamat.cu -lcublas -lcuda -lcudart

clean:
	@rm *.linkinfo *.pyc *.so
