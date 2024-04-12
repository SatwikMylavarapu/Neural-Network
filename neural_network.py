import numpy as np
from pathlib import Path

def load_data_small():
    """ Load small training and validation dataset

        Returns a tuple of length 4 with the following objects:
        X_train: An N_train-x-M ndarray containing the training data (N_train examples, M features each)
        y_train: An N_train-x-1 ndarray contraining the labels
        X_val: An N_val-x-M ndarray containing the training data (N_val examples, M features each)
        y_val: An N_val-x-1 ndarray contraining the labels
    """
    script_dir = Path("/Users/Spring24/ML/ASSIGNMENT1 QUES/Project-2/project2-programming")
    train_all = np.loadtxt(f'{script_dir}/data/smallTrain.csv', dtype=int, delimiter=',')
    valid_all = np.loadtxt(f'{script_dir}/data/smallValidation.csv', dtype=int, delimiter=',')

    X_train = train_all[:, 1:]
    y_train = train_all[:, 0]
    X_val = valid_all[:, 1:]
    y_val = valid_all[:, 0]

    return (X_train, y_train, X_val, y_val)


def load_data_medium():
    """ Load medium training and validation dataset

        Returns a tuple of length 4 with the following objects:
        X_train: An N_train-x-M ndarray containing the training data (N_train examples, M features each)
        y_train: An N_train-x-1 ndarray contraining the labels
        X_val: An N_val-x-M ndarray containing the training data (N_val examples, M features each)
        y_val: An N_val-x-1 ndarray contraining the labels
    """
    script_dir = Path("/Users/Spring24/ML/ASSIGNMENT1 QUES/Project-2/project2-programming")
    train_all = np.loadtxt(f'{script_dir}/data/mediumTrain.csv', dtype=int, delimiter=',')
    valid_all = np.loadtxt(f'{script_dir}/data/mediumValidation.csv', dtype=int, delimiter=',')

    X_train = train_all[:, 1:]
    y_train = train_all[:, 0]
    X_val = valid_all[:, 1:]
    y_val = valid_all[:, 0]

    return (X_train, y_train, X_val, y_val)


def load_data_large():
    """ Load large training and validation dataset

        Returns a tuple of length 4 with the following objects:
        X_train: An N_train-x-M ndarray containing the training data (N_train examples, M features each)
        y_train: An N_train-x-1 ndarray contraining the labels
        X_val: An N_val-x-M ndarray containing the training data (N_val examples, M features each)
        y_val: An N_val-x-1 ndarray contraining the labels
    """
    script_dir = Path("/Users/Spring24/ML/ASSIGNMENT1 QUES/Project-2/project2-programming")
    train_all = np.loadtxt(f'{script_dir}/data/largeTrain.csv', dtype=int, delimiter=',')
    valid_all = np.loadtxt(f'{script_dir}/data/largeValidation.csv', dtype=int, delimiter=',')

    X_train = train_all[:, 1:]
    y_train = train_all[:, 0]
    X_val = valid_all[:, 1:]
    y_val = valid_all[:, 0]

    return (X_train, y_train, X_val, y_val)



def linearForward(input, p):
    """
    :param input: input vector (column vector) WITH bias feature added
    :param p: parameter matrix (alpha/beta) WITH bias parameter added
    :return: output vector
    """
    output = np.dot(p, input)
    return output

def sigmoidForward(a):
    """
    :param a: input vector WITH bias feature added
    """
    output = 1 / (1 + np.exp(-a))
    return output


def softmaxForward(b):
    """
    :param b: input vector WITH bias feature added
    """
    exp_values = np.exp(b)
    exp_values_sum = np.sum(exp_values)
    sol = exp_values/exp_values_sum
    return sol

def crossEntropyForward(hot_y, y_hat):
    """
    :param hot_y: 1-hot vector for true label
    :param y_hat: vector of probabilistic distribution for predicted label
    :return: float
    """
    loss = -np.mean(np.sum(hot_y * np.log(y_hat)))
    return loss

def one_hot_encoder(y,num_classes):

    ohe = np.zeros(num_classes)
    ohe[y]=1
    return ohe

def NNForward(x, y, alpha, beta):
    """
    :param x: input data (column vector) WITH bias feature added
    :param y: input (true) labels
    :param alpha: alpha WITH bias parameter added
    :param beta: alpha WITH bias parameter added
    :return: all intermediate quantities x, a, z, b, y, J #refer to writeup for details
    TIP: Check on your dimensions. Did you make sure all bias features are added?
    """
   
    a=linearForward(x,alpha)
    z=np.insert(sigmoidForward(a),0,1).reshape(-1,1)
    b=linearForward(z,beta)

    y_hat=softmaxForward(b)

    y=one_hot_encoder(y,len(y_hat)).reshape(-1,1)
    J=crossEntropyForward(y,y_hat)
    
    return x,a,z,b,y_hat,J

def softmaxBackward(hot_y, y_hat):
    """
    :param hot_y: 1-hot vector for true label
    :param y_hat: vector of probabilistic distribution for predicted label
    """
    
    return y_hat-hot_y.reshape(-1,1)

def linearBackward(prev, p, grad_curr):
    """
    :param prev: previous layer WITH bias feature
    :param p: parameter matrix (alpha/beta) WITH bias parameter
    :param grad_curr: gradients for current layer
    :return:
        - grad_param: gradients for parameter matrix (alpha/beta)
        - grad_prevl: gradients for previous layer
    TIP: Check your dimensions.
    """
    grad_param = np.dot(grad_curr, prev.reshape(-1,1).T)
    grad_prev = np.dot(p.T, grad_curr)
    return grad_param, grad_prev


def sigmoidBackward(curr, grad_curr):
    """
    :param curr: current layer WITH bias feature
    :param grad_curr: gradients for current layer
    :return: grad_prev: gradients for previous layer
    TIP: Check your dimensions
    """
    sigmoid_grad = curr * (1 - curr)
    grad_prev = grad_curr * sigmoid_grad
    return grad_prev


def NNBackward(x, y, alpha, beta, z, y_hat):
    """
    :param x: input data (column vector) WITH bias feature added
    :param y: input (true) labels
    :param alpha: alpha WITH bias parameter added
    :param beta: alpha WITH bias parameter added
    :param z: z as per writeup
    :param y_hat: vector of probabilistic distribution for predicted label
    :return:
        - grad_alpha: gradients for alpha
        - grad_beta: gradients for beta
        - g_b: gradients for layer b (softmaxBackward)
        - g_z: gradients for layer z (linearBackward)
        - g_a: gradients for layer a (sigmoidBackward)
    """
    # Convert y to one-hot encoding
    y_one_hot =one_hot_encoder(y,10)
    g_y_hat =softmaxBackward(y_one_hot,y_hat)
    grad_beta, g_b = linearBackward(z,beta,g_y_hat)
    g_a =sigmoidBackward(z,g_b)
    g_b=g_b[1:,:]
    g_a=g_a[1:,:]
    grad_alpha, _ =linearBackward(x,alpha,g_a)

    return grad_alpha,grad_beta,g_y_hat,g_b,g_a

def SGD(tr_x, tr_y, valid_x, valid_y, hidden_units, num_epoch, init_flag, learning_rate):
    """
    :param tr_x: Training data input (size N_train x M)
    :param tr_y: Training labels (size N_train x 1)
    :param tst_x: Validation data input (size N_valid x M)
    :param tst_y: Validation labels (size N_valid x 1)
    :param hidden_units: Number of hidden units
    :param num_epoch: Number of epochs
    :param init_flag:
        - True: Initialize weights to random values in Uniform[-0.1, 0.1], bias to 0
        - False: Initialize weights and bias to 0
    :param learning_rate: Learning rate
    :return:
        - alpha weights
        - beta weights
        - train_entropy (length num_epochs): mean cross-entropy loss for training data for each epoch
        - valid_entropy (length num_epochs): mean cross-entropy loss for validation data for each epoch
    """
    
    
    # Initialize weights
    train_entropy,test_entropy=[],[]

    if init_flag:
        alpha=np.random.uniform(-0.1,0.1,size=(hidden_units,tr_x.shape[1]+1))
        beta=np.random.uniform(-0.1,0.1,size=(10,hidden_units+1))
    else:
        alpha=np.zeros((hidden_units,tr_x.shape[1]+1))
        beta=np.zeros((10,hidden_units+1))

        alpha[:, 0] = 0
        beta[:,0]=0

    for epoch in range(num_epoch):

        # Itarate over training data
        for epoch in range(tr_x.shape[0]):

            x,a,z,b,y_hat,J=NNForward(np.insert(tr_x[epoch],0,1),tr_y[epoch],alpha,beta)
            grad_alpha, grad_beta, g_y_hat, g_b_no_bias, g_a = NNBackward(x,tr_y[epoch],alpha,beta,z,y_hat)
            alpha-=learning_rate*grad_alpha
            beta-=learning_rate*grad_beta
    
        temp1=[NNForward(np.insert(tr_x[i], 0, 1), tr_y[i], alpha, beta)[-1] for i in range(len(tr_x))]
        train_entropy.append(np.mean(temp1))

        temp2=[NNForward(np.insert(valid_x[i], 0, 1), valid_y[i], alpha, beta)[-1] for i in range(len(valid_y))]
        test_entropy.append(np.mean(temp2))

    return alpha,beta,train_entropy,test_entropy

def prediction(tr_x, tr_y, valid_x, valid_y, tr_alpha, tr_beta):
    """
    :param tr_x: Training data input (size N_train x M)
    :param tr_y: Training labels (size N_train x 1)
    :param valid_x: Validation data input (size N_valid x M)
    :param valid_y: Validation labels (size N-valid x 1)
    :param tr_alpha: Alpha weights WITH bias
    :param tr_beta: Beta weights WITH bias
    :return:
        - train_error: training error rate (float)
        - valid_error: validation error rate (float)
        - y_hat_train: predicted labels for training data
        - y_hat_valid: predicted labels for validation data
    """
    bias_column = np.ones((tr_x.shape[0], 1))
    tr_x = np.hstack((bias_column, tr_x))
    y_hat_train_labels = []
    for i in range(len(tr_x)):
            # Forward pass
            # print(tr_x.shape)
            x = tr_x[i]
            y = tr_y[i]

            x = x.reshape(-1, 1)

            x, a, z, b, y_hat, J = NNForward(x, y, tr_alpha, tr_beta)

            y_hat_train_labels.append(np.argmax(y_hat))

    bias_column = np.ones((valid_x.shape[0], 1))
    valid_x = np.hstack((bias_column, valid_x))
    y_hat_valid_labels = []
    # Forward pass on validation data
    for i in range(len(valid_x)):
            x = valid_x[i]
            y = valid_y[i]

            x = x.reshape(-1, 1)
            _, _, _, _, y_hat_valid, J_valid = NNForward(x, y, tr_alpha, tr_beta)
            
            y_hat_valid_labels.append(np.argmax(y_hat_valid))


    y_hat_train_labels = np.array(y_hat_train_labels)
    # Calculate training error rate
    train_error = np.mean(y_hat_train_labels != tr_y.flatten())

    y_hat_valid_labels = np.array(y_hat_valid_labels)
    # Calculate validation error rate
    valid_error = np.mean(y_hat_valid_labels != valid_y.flatten())

    return train_error, valid_error, y_hat_train_labels, y_hat_valid_labels


def train_and_valid(X_train, y_train, X_val, y_val, num_epoch, num_hidden, init_flag, learning_rate):
    """ Main function to train and validate your neural network implementation.

        X_train: Training input in N_train-x-M numpy nd array. Each value is binary, in {0,1}.
        y_train: Training labels in N_train-x-1 numpy nd array. Each value is in {0,1,...,K-1},
            where K is the number of classes.
        X_val: Validation input in N_val-x-M numpy nd array. Each value is binary, in {0,1}.
        y_val: Validation labels in N_val-x-1 numpy nd array. Each value is in {0,1,...,K-1},
            where K is the number of classes.
        num_epoch: Positive integer representing the number of epochs to train (i.e. number of
            loops through the training data).
        num_hidden: Positive integer representing the number of hidden units.
        init_flag: Boolean value of True/False
        - True: Initialize weights to random values in Uniform[-0.1, 0.1], bias to 0
        - False: Initialize weights and bias to 0
        learning_rate: Float value specifying the learning rate for SGD.

        RETURNS: a tuple of the following six objects, in order:
        loss_per_epoch_train (length num_epochs): A list of float values containing the mean cross entropy on training data after each SGD epoch
        loss_per_epoch_val (length num_epochs): A list of float values containing the mean cross entropy on validation data after each SGD epoch
        err_train: Float value containing the training error after training (equivalent to 1.0 - accuracy rate)
        err_val: Float value containing the validation error after training (equivalent to 1.0 - accuracy rate)
        y_hat_train: A list of integers representing the predicted labels for training data
        y_hat_val: A list of integers representing the predicted labels for validation data
    """
    ### YOUR CODE HERE'
    alpha, beta, train_entropy, valid_entropy = SGD(X_train, y_train, X_val, y_val, num_hidden, num_epoch, init_flag, learning_rate)

    # Predict labels for training and validation data
    train_error, valid_error, y_hat_train, y_hat_val = prediction(X_train, y_train, X_val, y_val, alpha, beta)

    # Compute mean cross entropy on training and validation data after each epoch
    # loss_per_epoch_train = [crossEntropyForward(y_train, y_hat) / len(y_train) for y_hat in y_hat_train]
    # loss_per_epoch_val = [crossEntropyForward(y_val, y_hat) / len(y_val) for y_hat in y_hat_val]

    # Compute training and validation errors
    # err_train = train_error
    # err_val = valid_error

    return train_entropy, valid_entropy, train_error, valid_error, y_hat_train,y_hat_val