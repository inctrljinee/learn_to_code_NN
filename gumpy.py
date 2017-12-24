import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import pycuda.driver as drv
import numpy as np

import skcuda.linalg as culinalg
import skcuda.misc as cumisc

import cProfile as profile
from timeit import default_timer as timer

culinalg.init()


def gt(x):
    return gpuarray.to_gpu(x)

def dot(a, b):
    a_gpu = gt(a)
    b_gpu = gt(b)
    return culinalg.dot(a_gpu, b_gpu) #scikit-cuda wrapper takes care of the operation!

def array(x):
    return gpuarray.to_gpu(np.array(x))

if __name__ == "__main__":
    
    M = 10
    N = 10

    ac = np.random.rand(M, N) 
    bc = np.random.rand(M, N)

    cc = np.dot(ac, bc)
    cg = gdot(ac, bc)

