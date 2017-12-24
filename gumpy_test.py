import gumpy as gp
import numpy as np


M = 10
N = 10

ac = np.random.rand(M, N) 
bc = np.random.rand(M, N)

cc = np.dot(ac, bc)
cg = gp.dot(ac, bc)



