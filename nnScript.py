import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt


def initializeWeights(n_in, n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
       
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""

    epsilon = sqrt(6) / sqrt(n_in + n_out + 1)
    W = (np.random.rand(n_out, n_in + 1) * 2 * epsilon) - epsilon
    return W


def sigmoid(z):
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""

    #I use this equation: S=1/(1+math.exp^(-z))  , but I am not sure is -z correct for vector and matrix--YC

    return 1./(1.+np.exp(-1.0*z)) # your code here

def get_one_hot(targets, nb_classes):
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape)+[nb_classes])

def preprocess():
    """ Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains 
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data 
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the 
       training set
     test_data: matrix of training set. Each row of test_data contains 
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set

     Some suggestions for preprocessing step:
     - feature selection"""

    mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary

    # Split the training sets into two sets of 50000 randomly sampled training examples and 10000 validation examples. 
    # Your code here.
    E_I=[]                 #This list is a list of 2-elements list, 1st element is rows in the train matrix, 2nd element is the label of that row
    T_I=[]                 #This list is a list of 2-elements list, 1st element is rows in the test matrix, 2nd element is the label of that row
    for i in range(10):
        trainname='train'+str(i)
        trainmatrix = mat.get(trainname)
        for trainrow in trainmatrix:
            E_I.append([trainrow,i])

        testname='test'+str(i)
        testmatrix=mat.get(testname)
        for testrow in testmatrix:
            T_I.append([testrow,i])

    RE_I = np.random.permutation(E_I)#order E_I randomly and split it 50,000:10,000
    S=RE_I[:50000]
    V=RE_I[50000:]



    #Up to this step, both S and V are a list of 2-elements list:[[row, label of row]...] We need to save row and label differently
    #Sdata Vdata for data
    #Slabel Vlabel for label


    Sdata=[]
    Slabel=[]
    for i in S:
        Sdata.append(i[0])
        Slabel.append(i[1])
    train_data= np.array(Sdata,dtype=float)                          #This is the original one without feature selection, size is 784.
    train_label=np.array(Slabel,dtype=float)

    #2019/4/8
    #Use these code to replace line 123-136, I move Feature Selection to here, then we can directly delete those ignored cols for Vdata and Tdata
    y = np.std(train_data,axis=0)       #I found use np.array to calculate col's std is faster than list
    F=[]
    for i in range(len(y)):
        if y[i]==0:
            F.append(i)
    print(len(F),F)
    #Up to this step, we already get a train_data (size 784) and a list of cols need to be ignored (size around 70)
    #Our original method is rewrite a list and remove cols in F , BUT this time we will directly delete this columns when we create V and T
    #At first create a int list[0,1,2...,783] this is the original index list, remove ignored cols index from index_list
    index_list=[]
    for i in range(len(Sdata[0])):
        if i not in F:
            index_list.append(i)
    #print(len(index_list))
    new_order_index = np.array(index_list)

    train_data = train_data[:, new_order_index] #np.array[:,[order]] can make a new np.array with the order we want

    Vdata = []
    Vlabel = []
    for i in V:
        Vlabel.append(i[1])
        Vdata.append(i[0])
    validation_data = np.array(Vdata, dtype=float)
    validation_data = validation_data[:,new_order_index]
    validation_label = np.array(Vlabel, dtype=float)


    Tdata =[]
    Tlabel = []
    for i in T_I:
        Tdata.append(i[0])
        Tlabel.append(i[1])
    test_data = np.array(Tdata, dtype=float)
    test_data = test_data[:,new_order_index]
    test_label = np.array(Tlabel, dtype=float)


    '''
    # Feature selection
    # Your code here.
    #2019/4/8 Move this part to line #94-#101
    F=[]                               #This list is used to save those columns whose standard deviation=0, means every row's value on that column are same,
    for i in range(len(Sdata[0])):
        features=[]
        for s in Sdata:
            features.append(s[i])
        standard_deviation=np.std(features)
        #print(standard_deviation)
        if standard_deviation==0:
            F.append(i)
    print(len(F),F)                          # printout the F, it size is 70
    



    FS=[]                   #Feature selection Sdata: Sdata is a list of row, for each row, rewrite a new row same with it except those columns in F.
    for s in Sdata:
        fs=[]
        for i in range(len(s)):
            if i not in F:
                fs.append(s[i])
        FS.append(fs)
    train_data = np.array(FS, dtype=float)

    
    #Do samething on Vdata and Tdata
    FV = []                         # Feature selection Vdata
    for v in Vdata:
        fv = []
        for i in range(len(v)):
            if i not in F:
                fv.append(v[i])
        FV.append(fv)
    validation_data = np.array(FV, dtype=float)
    

    FT = []                         # Feature selection Tdata
    for t in Tdata:
        ft = []
        for i in range(len(t)):
            if i not in F:
                ft.append(t[i])
        FT.append(ft)
    test_data = np.array(FT, dtype=float)
    '''


    train_data/=255
    validation_data/=255
    test_data/=255

    print('preprocess done')

    return train_data, train_label, validation_data, validation_label, test_data, test_label


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
    #print(a1)
    a1 = sigmoid(a1)
    a1_with_bias = np.ones((a1.shape[0], a1.shape[1] + 1))
    a1_with_bias[:, :-1] = a1
    a2 = np.dot(a1_with_bias, w2.T)
    #print(a2)
    a2 = sigmoid(a2)
    #     print("a2 shape = ")
    #     print(a2.shape)
    #     training_label = int(training_label)
    train_label_hot = get_one_hot(training_label.astype(int), 10)

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


"""**************Neural Network Script Starts here********************************"""

train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()

#  Train Neural Network

# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]

# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 50

# set the number of nodes in output unit
n_class = 10

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden)
initial_w2 = initializeWeights(n_hidden, n_class)

# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()), 0)

# set the regularization hyper-parameter
lambdaval = 0

args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

# Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example

opts = {'maxiter': 50}  # Preferred value.

nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)

# In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
# and nnObjGradient. Check documentation for this function before you proceed.
# nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)


# Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

# Test the computed parameters

predicted_label = nnPredict(w1, w2, train_data)

# find the accuracy on Training Dataset

print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')

predicted_label = nnPredict(w1, w2, validation_data)

# find the accuracy on Validation Dataset

print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')

predicted_label = nnPredict(w1, w2, test_data)

# find the accuracy on Validation Dataset

print('\n Test set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')
