import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import pycuda.driver as drv
import numpy as np

import skcuda.linalg as culinalg
import skcuda.misc as cumisc

import cProfile as profile
from timeit import default_timer as timer


import time

culinalg.init()

# Double precision is only supported by devices with compute
# capability >= 1.3:
import string
demo_types = [np.float32, np.complex64]
if cumisc.get_compute_capability(pycuda.autoinit.device) >= 1.3:
    demo_types.extend([np.float64, np.complex128])

n = 4096 *2

def matrix_multiply_GPU():
    for t in demo_types:
        print('Testing matrix multiplication for type ' + str(np.dtype(t)))
        
        #Ensure that object type is correct
        if np.iscomplexobj(t()):
            a = np.asarray(np.random.rand(n,n)+1j*np.random.rand(n,n), t)
            b = np.asarray(np.random.rand(n,n)+1j*np.random.rand(n,n), t)
        else:
            a = np.asarray(np.random.rand(n,n), t)
            b = np.asarray(np.random.rand(n,n), t)

        #Transferring data to GPU
        a_gpu = gpuarray.to_gpu(a)
        b_gpu = gpuarray.to_gpu(b)

        c_gpu = culinalg.dot(a_gpu, b_gpu) #scikit-cuda wrapper takes care of the operation!
        
def matrix_multiply_CPU():
    for t in demo_types:
        print('Testing matrix multiplication for type ' + str(np.dtype(t)))
        if np.iscomplexobj(t()):
            a = np.asarray(np.random.rand(n,n)+1j*np.random.rand(n,n), t)
            b = np.asarray(np.random.rand(n,n)+1j*np.random.rand(n,n), t)
        else:
            a = np.asarray(np.random.rand(n,n), t)
            b = np.asarray(np.random.rand(n,n), t)

        c = np.dot(a, b)

def matmul(a, b):
        
        #Ensure that object type is correct
        # if np.iscomplexobj(t()):
            # a = np.asarray(np.random.rand(n,n)+1j*np.random.rand(n,n), t)
            # b = np.asarray(np.random.rand(n,n)+1j*np.random.rand(n,n), t)
        # else:
            # a = np.asarray(np.random.rand(n,n), t)
            # b = np.asarray(np.random.rand(n,n), t)
    
    #Transferring data to GPU
    a_gpu = gpuarray.to_gpu(a)
    b_gpu = gpuarray.to_gpu(b)

    c_gpu = culinalg.dot(a_gpu, b_gpu) #scikit-cuda wrapper takes care of the operation!
    return c_gpu

if __name__ == '__main__':
    import timeit

    #     profile.run("matrix_multiply_GPU()", sort="time")
    #     profile.run("matrix_multiply_CPU()", sort="time")
    # print('\nPerformance for performing matrix multiplicate of 4096 x 4096 matrices of several data types on GPU:')
    # ts = timer()
    # matrix_multiply_GPU()
    # te = timer()
    # elapsed = te - ts
    # fmt = '%20s: %s'
    # print(fmt % ('time elapsed', '%.3fs' % (te - ts)))


    # print('\nPerformance for performing matrix multiplicate of 4096 x 4096 matrices of several data types on CPU:')
    # ts = timer()
    # matrix_multiply_CPU()
    # te = timer()
    # elapsed = te - ts
    # fmt = '%20s: %s'
    # print(fmt % ('time elapsed', '%.3fs' % (te - ts)))
   
    N = 20
    M = 200
    g_time = time.time()
    for k in range(N):
        a = np.random.rand(M, M) 
        b = np.random.rand(M, M)
        c_gpu = matmul(a, b)
        gm_time = time.time() - g_time
        print('ETA: ', gm_time)
    g_time = time.time() - g_time

    c_time = time.time()
    for k in range(N):
        a = np.random.rand(M, M) 
        b = np.random.rand(M, M)
        c_cpu = np.dot(a, b)
        cm_time = time.time() - c_time
        print('ETA: ', cm_time)
    
    c_time = time.time() - c_time
    print('GPU time : %f, CPU time %f'%(g_time, c_time))
