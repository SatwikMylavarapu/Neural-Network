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
    script_dir = Path("/Users/Spring24/ML/'ASSIGNMENT1 QUES'/Project-2/project2-programming")
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
    script_dir = Path("/Users/Spring24/ML/'ASSIGNMENT1 QUES'/Project-2/project2-programming")
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
    script_dir = Path("/Users/Spring24/ML/'ASSIGNMENT1 QUES'/Project-2/project2-programming")
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
    output = np.dot(p.T, input)
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
    exp_b = np.exp(b - np.max(b))
    return exp_b / np.sum(exp_b)


def crossEntropyForward(hot_y, y_hat):
    """
    :param hot_y: 1-hot vector for true label
    :param y_hat: vector of probabilistic distribution for predicted label
    :return: float
    """
    return -np.sum(hot_y * np.log(y_hat))


def NNForward(x, y, alpha, beta):
    """
    :param x: input data (column vector) WITH bias feature added
    :param y: input (true) labels
    :param alpha: alpha WITH bias parameter added
    :param beta: alpha WITH bias parameter added
    :return: all intermediate quantities x, a, z, b, y, J #refer to writeup for details
    TIP: Check on your dimensions. Did you make sure all bias features are added?
    """
    # Convert y to one-hot encoding
    hot_y = np.zeros((len(y), len(np.unique(y))))
    hot_y[np.arange(len(y)), y] = 1

    # Add bias feature to input x
    x_with_bias = np.vstack([x, [1]])  # Adding bias feature

    # Apply linear transformation from input to hidden layer
    a = np.dot(alpha, x_with_bias)

    # Apply sigmoid activation function
    z = sigmoidForward(a)

    # Add bias feature to hidden layer output
    z_with_bias = np.vstack([z, [1]])  # Adding bias feature

    # Apply linear transformation from hidden to output layer
    b = np.dot(beta, z_with_bias)

    # Apply softmax activation function
    y_hat = softmaxForward(b)

    # Compute the cross-entropy loss
    J = crossEntropyForward(hot_y, y_hat)

    return x_with_bias, a, z_with_bias, b, y_hat, J


def softmaxBackward(hot_y, y_hat):
    """
    :param hot_y: 1-hot vector for true label
    :param y_hat: vector of probabilistic distribution for predicted label
    """
    return y_hat - hot_y


def linearBackward(prev, p, grad_curr):
    """
    :param prev: previous layer WITH bias feature
    :param p: parameter matrix (alpha/beta) WITH bias parameter
    :param grad_curr: gradients for current layer
    :return:
        - grad_param: gradients for parameter matrix (alpha/beta)
        - grad_prev: gradients for previous layer
    TIP: Check your dimensions.
    """
    grad_param = np.dot(prev, grad_curr.T)
    grad_prev = np.dot(p, grad_curr)
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
    hot_y = np.zeros((len(y), len(np.unique(y))))
    hot_y[np.arange(len(y)), y] = 1

    # Gradient of Cross Entropy Loss w.r.t. y_hat
    g_y_hat = softmaxBackward(hot_y, y_hat)

    # Gradient of Loss w.r.t. beta (Weights from hidden to output layer)
    grad_beta, g_b = linearBackward(z, beta, g_y_hat)

    # Gradient of Loss w.r.t. activation before sigmoid (a)
    g_a = np.dot(beta[:, :-1].T, g_b)

    # Gradient of Loss w.r.t. alpha (Weights from input to hidden layer)
    grad_alpha, g_z = linearBackward(x, alpha, g_a)

    return grad_alpha, grad_beta, g_b, g_z, g_a


def SGD(tr_x, tr_y, valid_x, valid_y, hidden_units, num_epoch, init_flag, learning_rate):
    # Initialize weights and biases
    alpha = np.random.uniform(-0.1, 0.1, (hidden_units, tr_x.shape[1] + 1)) if init_flag else np.zeros((hidden_units, tr_x.shape[1] + 1))
    beta = np.random.uniform(-0.1, 0.1, (len(np.unique(tr_y)), hidden_units + 1)) if init_flag else np.zeros((len(np.unique(tr_y)), hidden_units + 1))

    train_entropy = []
    valid_entropy = []

    for epoch in range(num_epoch):
        # Shuffle training data
        indices = np.random.permutation(len(tr_x))
        tr_x_shuffled = tr_x[indices]
        tr_y_shuffled = tr_y[indices]

        total_loss_train = 0

        for i in range(len(tr_x_shuffled)):
            # Forward pass
            x = tr_x_shuffled[i]
            y = tr_y_shuffled[i]
            x_with_bias = np.hstack([x, 1])  # Add bias feature
            x_with_bias = x_with_bias.reshape(-1, 1)  # Convert to column vector
            y = np.array([y])

            x, a, z, b, y_hat, J = NNForward(x_with_bias, y, alpha, beta)

            # Backward pass
            grad_alpha, grad_beta, _, _, _ = NNBackward(x_with_bias, y, alpha, beta, z, y_hat)

            # Update weights
            alpha -= learning_rate * grad_alpha
            beta -= learning_rate * grad_beta

            total_loss_train += J

        # Calculate mean loss for training data
        mean_loss_train = total_loss_train / len(tr_x_shuffled)
        train_entropy.append(mean_loss_train)

        # Calculate mean loss for validation data
        _, _, _, _, _, J_valid = NNForward(valid_x.T, valid_y, alpha, beta)
        mean_loss_valid = J_valid / len(valid_x)
        valid_entropy.append(mean_loss_valid)

    return alpha, beta, train_entropy, valid_entropy

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
    # Forward pass on training data
    _, _, _, _, y_hat_train, _ = NNForward(tr_x.T, tr_y, tr_alpha, tr_beta)
    y_hat_train_labels = np.argmax(y_hat_train, axis=1)

    # Forward pass on validation data
    _, _, _, _, y_hat_valid, _ = NNForward(valid_x.T, valid_y, tr_alpha, tr_beta)
    y_hat_valid_labels = np.argmax(y_hat_valid, axis=1)

    # Calculate training error rate
    train_error = np.mean(y_hat_train_labels != tr_y.flatten())

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
    # Train the neural network using SGD
    alpha, beta, train_entropy, valid_entropy = SGD(X_train, y_train, X_val, y_val, num_hidden, num_epoch, init_flag, learning_rate)

    # Predict labels for training and validation data
    train_error, valid_error, y_hat_train, y_hat_val = prediction(X_train, y_train, X_val, y_val, alpha, beta)

    # Compute mean cross entropy on training and validation data after each epoch
    loss_per_epoch_train = [crossEntropyForward(y_train, y_hat) / len(y_train) for y_hat in y_hat_train]
    loss_per_epoch_val = [crossEntropyForward(y_val, y_hat) / len(y_val) for y_hat in y_hat_val]

    # Compute training and validation errors
    err_train = train_error
    err_val = valid_error

    return loss_per_epoch_train, loss_per_epoch_val, err_train, err_val, y_hat_train, y_hat_val