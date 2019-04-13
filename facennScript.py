'''
Comparing single layer MLP with deep MLP (using TensorFlow)
'''

import numpy as np
import pickle
from math import sqrt
from scipy.optimize import minimize

def get_one_hot(targets, nb_classes):
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape)+[nb_classes])

# Do not change this
def initializeWeights(n_in,n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
                            
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""
    epsilon = sqrt(6) / sqrt(n_in + n_out + 1);
    W = (np.random.rand(n_out, n_in + 1)*2* epsilon) - epsilon;
    return W



# Replace this with your sigmoid implementation
def sigmoid(z):
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""

    #I use this equation: S=1/(1+math.exp^(-z))  , but I am not sure is -z correct for vector and matrix--YC

    return 1./(1.+np.exp(-1.0*z)) # your code here
    
# Replace this with your nnObjFunction implementation
def nnObjFunction(params, *args):
    """% nnObjFunction computes the value of objective function (negative log
    %   likelihood error function with regularization) given the parameters
    %   of Neural Networks, thetraining data, their corresponding training
    %   labels and lambda - regularization hyper-parameter.

    % Input:
    % params: vector of weights of 2 matrices w1 (weights of connections from
    %     input layer to hidden layer) and w2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not include the bias node)
    % n_hidden: number of node in hidden layer (not include the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % training_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % training_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.

    % Output:
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % reshape 'params' vector into 2 matrices of weight w1 and w2
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden
    %     layer to unit i in output layer."""

    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args

    w1 = np.array(params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1))))
    w2 = np.array(params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1))))
    training_data = np.array(training_data)

    # Your code here
    #     print("w1 shape = ")
    #     print(w1.shape)
    #     print("w2 shape = ")
    #     print(w2.shape)
    training_data_with_bias = np.ones((training_data.shape[0], n_input + 1))
    training_data_with_bias[:, :-1] = training_data
    #     print("training data with bias shape")
    #     print(training_data_with_bias.shape)
    a1 = np.dot(training_data_with_bias, w1.T)
    # print(a1)
    a1 = sigmoid(a1)
    a1_with_bias = np.ones((a1.shape[0], a1.shape[1] + 1))
    a1_with_bias[:, :-1] = a1
    a2 = np.dot(a1_with_bias, w2.T)
    # print(a2)
    a2 = sigmoid(a2)
    #     print("a2 shape = ")
    #     print(a2.shape)
    #     training_label = int(training_label)
    train_label_hot = get_one_hot(training_label.astype(int), 2)

    obj_val = np.sum(np.multiply(train_label_hot, np.log(a2)) + np.multiply((1.0 - train_label_hot), np.log(1.0 - a2)))
    obj_val /= training_data.shape[0] * (-1)

    obj_val_regularization = (lambdaval / (2 * training_label.shape[0])) * (
            np.sum(np.square(w1)) + np.sum(np.square(w2)))
    obj_val += obj_val_regularization

    #     grad_w1 = np.zeros((w1.shape[0], w1.shape[1]))
    #     grad_w2 = np.zeros((w2.shape[0], w2.shape[1]))

    dl = (a2 - train_label_hot)
    grad_w2 = np.dot(dl.T, a1_with_bias)
    grad_w2 += (lambdaval * w2)
    grad_w2 /= training_data.shape[0]
    temp = np.dot(dl, w2)
    temp = ((1 - a1_with_bias) * a1_with_bias) * temp
    grad_w1 = np.dot(temp.T, training_data_with_bias)
    grad_w1 = np.delete(grad_w1, n_hidden, 0)
    grad_w1 += (lambdaval * w1)
    grad_w1 /= training_data.shape[0]
    #     print("grad_wx shape = ")
    #     print(grad_w1.shape)
    #     print(grad_w2.shape)

    # Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    # you would use code similar to the one below to create a flat array
    obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()), 0)

    return (obj_val, obj_grad)
    
# Replace this with your nnPredict implementation
def nnPredict(w1, w2, data):
    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
    % Network.

    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit i in input
    %     layer to unit j in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit i in input
    %     layer to unit j in hidden layer.
    % data: matrix of data. Each row of this matrix represents the feature
    %       vector of a particular image

    % Output:
    % label: a column vector of predicted labels"""

    # Your code here
    data_with_bias = np.ones((data.shape[0], data.shape[1] + 1))
    data_with_bias[:, :-1] = data

    a1 = sigmoid(np.dot(data_with_bias, w1.T))
    a1_with_bias = np.ones((a1.shape[0], a1.shape[1] + 1))
    a1_with_bias[:, :-1] = a1
    a2 = sigmoid(np.dot(a1_with_bias, w2.T))

    labels = np.argmax(a2, axis=1)

    return labels

# Do not change this
def preprocess():
    pickle_obj = pickle.load(file=open('face_all.pickle', 'rb'))
    features = pickle_obj['Features']
    labels = pickle_obj['Labels']
    train_x = features[0:21100] / 255
    valid_x = features[21100:23765] / 255
    test_x = features[23765:] / 255

    labels = labels[0]
    train_y = labels[0:21100]
    valid_y = labels[21100:23765]
    test_y = labels[23765:]
    return train_x, train_y, valid_x, valid_y, test_x, test_y

"""**************Neural Network Script Starts here********************************"""
train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()
#  Train Neural Network
# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]
# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 50
# set the number of nodes in output unit
n_class = 2

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden);
initial_w2 = initializeWeights(n_hidden, n_class);
# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()),0)
# set the regularization hyper-parameter
lambdaval = 5;
args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

#Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example
opts = {'maxiter' :50}    # Preferred value.

nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args,method='CG', options=opts)
params = nn_params.get('x')
#Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = params[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

#Test the computed parameters
predicted_label = nnPredict(w1,w2,train_data)
#find the accuracy on Training Dataset
print('\n Training set Accuracy:' + str(100*np.mean((predicted_label == train_label).astype(float))) + '%')
predicted_label = nnPredict(w1,w2,validation_data)
#find the accuracy on Validation Dataset
print('\n Validation set Accuracy:' + str(100*np.mean((predicted_label == validation_label).astype(float))) + '%')
predicted_label = nnPredict(w1,w2,test_data)
#find the accuracy on Validation Dataset
print('\n Test set Accuracy:' +  str(100*np.mean((predicted_label == test_label).astype(float))) + '%')
