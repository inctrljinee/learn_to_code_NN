import matplotlib
matplotlib.use('Agg')
from data_gen import *
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import sklearn.datasets
import sklearn.linear_model
import ipdb
import matplotlib.pyplot as plt
import numpy as np
# When it says "dx" it is actually "Err/dx". 

def data_gen(num):
    np.random.seed(0)
    X, y = sklearn.datasets.make_moons(num, noise=0.20)
    return X, y

def plot_decision_boundary_gif(pred_func, X, y, epoch):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
    plt.title("Decision Boundary for hidden layer size 3")
    # plt.show()
    plt.savefig('pic_folder/decision_boundary_' + str(epoch) + '.png')


class MultiplyGate:
    # W: (previous dim X next dim)
    def forward(self, W, X):
        return np.dot(X, W)
    
    def backward(self, W, X, Err):
        dW = np.dot(np.transpose(X), Err)
        dX = np.dot(Err, np.transpose(W))
        return dW, dX


class AddGate:
    def forward(self, X, b):
        return X + b

    def backward(self, X, b, Err):  # Err is error from the upper level layer.
        dX = Err * np.ones_like(X)
        # dx2 = Err * np.ones_like(x2)
        db = np.dot(np.ones((1, Err.shape[0]), dtype=np.float64), Err)
        return db, dX


class Sigmoid:
    def forward(self, X):
        return 1.0/(1.0 + np.exp(-X))
    def backward(self, X, E):
        output = self.forward(X)
        return (1.0 - output) * output * E


class Tanh:
    def forward(self, X):
        return np.tanh(X)
    
    def backward(self, X, top_diff):
        output = self.forward(X)
        return (1.0 - np.square(output)) * top_diff


class Softmax:
    def predict(self, X):
        exp_scores = np.exp(X)
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    def loss(self, X, y):
        num_examples = X.shape[0]
        probs = self.predict(X)
        correct_logprobs = -np.log(probs[range(num_examples), y])
        data_loss = np.sum(correct_logprobs)
        return (1./num_examples) * data_loss, probs

    def diff(self, X, y):
        num_examples = X.shape[0]
        probs = self.predict(X)
        probs[range(num_examples), y] -= 1
        return probs


class Model:
    def __init__(self, layers_dim):
        self.b = []
        self.W = []
        for i in range(len(layers_dim)-1):
            # Randomly initialize weights and biases.
            random_weights = np.random.randn(layers_dim[i], layers_dim[i+1]) / np.sqrt(layers_dim[i])
            random_biases = np.random.randn(layers_dim[i+1]).reshape(1, layers_dim[i+1])
            self.W.append(random_weights)
            self.b.append(random_biases)

    '''
    <Dimension Info>
    feed_input (=X) : batch_size * d_in
    mul = batch_size * d_out = (batch_size * d_in) X (d_in * d_out)
    add = batch_size * d_out
    activated_out = batch_size * d_out
    '''
    def feed_forward(self, X):
        mulGate = MultiplyGate()
        addGate = AddGate()
        activation_layer = Tanh()
        feed_input = X
        # ipdb.set_trace()
        forward = [(None, None, feed_input)]
        for i in range(len(self.W)):
            mul = mulGate.forward(self.W[i], feed_input)
            add = addGate.forward(mul, self.b[i])
            activated_out = activation_layer.forward(add)
            forward.append((mul, add, activated_out))
            feed_input = activated_out
        return activated_out, forward

    '''
    <Dimension Info>
    Err : batch_size * 
    '''
    def back_propagation(self, forward, Err, eps, reg_lambda):
        mulGate = MultiplyGate()
        addGate = AddGate()
        activation_layer = Tanh()
        for i in range(len(forward)-1, 0, -1):
            if i == 0:  # The last layer.
                dadd = activation_layer.backward(forward[i][1], Err)  # "add" and "Err" from the previous layer
            elif i > 0:  # Else than the last layer.
                dadd = activation_layer.backward(forward[i][1], Err)  # "add" and "Err" from the previous layer
            db, dmul = addGate.backward(forward[i][0], self.b[i-1], dadd)
            dW, Err = mulGate.backward(self.W[i-1], forward[i-1][2], dmul)
            dW += reg_lambda * self.W[i-1]
            self.b[i-1] += -eps * db
            self.W[i-1] += -eps * dW
            ipdb.set_trace()


    def calculate_loss(self, X, y):
        activated_out, forward = self.feed_forward(X)
        softmaxOutput = Softmax()
        return softmaxOutput.loss(activated_out, y) 


    def predict(self, X):
        activated_out, forward = self.feed_forward(X)
        softmaxOutput = Softmax()
        probs = softmaxOutput.predict(activated_out)
        return np.argmax(probs, axis=1)


    def batch_create(self, X, batch_size):
        batch_length = int(np.ceil(np.shape(X)[0]/batch_size))
        X_batch_list, y_batch_list = [], []
        for k in range(batch_length):
            if k < batch_length-1:
                X_batch_list.append(X[batch_size*(k):batch_size*(k+1), :])
                y_batch_list.append(y[batch_size*(k):batch_size*(k+1)])
            elif k == batch_length-1:
                X_batch_list.append(X[batch_size*(k):, :])
                y_batch_list.append(y[batch_size*(k):])

        return X_batch_list, y_batch_list

    def train(self, X, y, num_passes=20000, batch_size=200, eps=0.01, reg_lambda=0.01, print_loss=True):
        softmaxOutput = Softmax()
       
        X_batch_list, y_batch_list = self.batch_create(X, batch_size)
        for epoch in range(num_passes):

            for batch_index, X in enumerate(X_batch_list):
                y = y_batch_list[batch_index]
                # Feed Forward Network
                activated_out, forward = self.feed_forward(X)
                
                # Loss function 
                dtanh = softmaxOutput.diff(forward[len(forward)-1][2], y)
                
                # Back Propagation
                self.back_propagation(forward, dtanh, eps, reg_lambda)

                if print_loss and epoch % 1000 == 0:
                    y_est = self.predict(X)
                    print('Epoch', str(epoch), 'Batch:', str(batch_index), 'Train Acc. : ', np.sum(y_est==y)/float(len(y)))
                    print('Sum of label +: ', np.sum(y_est) )
                    loss, prob = self.calculate_loss(X, y)
                    print("Loss at iteration %i: %f" %(epoch, loss ))
                    # plot_decision_boundary_gif(lambda x: self.predict(x), X, y, epoch)


if __name__ == "__main__":
    
    X, y = data_gen(num=1245) 
    layers_dim = [2, 5, 2]  # input dimension, hidden layer with 3 dim, binary output

    model = Model(layers_dim)
    model.train(X, y, num_passes=30000, eps=0.001, reg_lambda=0.01, print_loss=True)

