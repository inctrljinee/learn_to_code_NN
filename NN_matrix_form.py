##############################################################
### 


import numpy as np
import time
import matplotlib.pyplot as plt
import math

start_time = time.time()

batch_size = 17
input_dim = 4
hidden1_dim = 5
hidden2_dim = 3
output_dim = 2

# batch_size = 17
# input_dim = 20
# hidden1_dim = 50
# hidden2_dim = 30
# output_dim = 10

x = np.random.randn(input_dim, batch_size)
y = np.random.randn(output_dim , batch_size)

w1 = np.random.randn(hidden1_dim, input_dim)
w2 = np.random.randn(hidden2_dim, hidden1_dim)
w3 = np.random.randn(output_dim , hidden2_dim)

learning_rate = 1e-6  
epoch = 500

loss = []
for t in range(epoch):
    
    # x: input layer and input values.
    w1_mm_x0 = w1.dot(x)
    hidden_layer_1_relued = np.maximum(w1_mm_x0, 0)
    x2 = hidden_layer_1_relued
    
    w2_mm_x1 = w2.dot(hidden_layer_1_relued)
    hidden_layer_2_relued = np.maximum(w2_mm_x1, 0)
    x3 = hidden_layer_2_relued

    w3_mm_x2 = w3.dot(hidden_layer_2_relued)
    y_pred = np.maximum(w3_mm_x2, 0)
    
    loss.append(np.square(y_pred - y).sum())
    print('for time:'+str(t) + 'loss:'+ str(loss[-1])) 
    
    # Gradient of dE/dW3
    delta_3 = (y_pred - y)
    delta_3[w3_mm_x2 < 0] = 0
    grad_w3 = delta_3.dot(hidden_layer_2_relued.T)

    # Gradiend of dE/dW2
    delta_2_relu = w3.T.dot(delta_3)
    delta_2 = delta_2_relu.copy()
    delta_2[w2_mm_x1 < 0] = 0
    grad_w2 = delta_2.dot(hidden_layer_1_relued.T)
    
    # Gradiend of dE/dW3
    delta_1_relu = w2.T.dot(delta_2)       
    delta_1 = delta_1_relu.copy()              
    delta_1[w1_mm_x0 < 0] = 0             
    grad_w1 = delta_1.dot(x.T)    
   
    w1 -= learning_rate * grad_w1  
    w2 -= learning_rate * grad_w2  
    w3 -= learning_rate * grad_w3

    if math.isnan(loss[-1]): 
        break

plt.figure()
plt.plot(loss)
plt.title('The loss function')
plt.show()
