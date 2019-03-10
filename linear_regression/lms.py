"""Least Mean Square Algorithm

Author: Cade Parkison
"""


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


test_data = pd.read_csv('concrete/test.csv', header=None)
train_data = pd.read_csv('concrete/train.csv', header=None)

# first 7 columns are features, last column (Slump) is output
columns = ['Cement', 'Slag', 'Fly ash', 'Water',
           'SP', 'Coarse Aggr', 'Fine Aggr', 'Slump']
features = columns[:-1]
output = columns[-1]

test_data.columns = columns
train_data.columns = columns

train_data_array = train_data.to_numpy()

A = train_data_array[:, :-1]
A = np.insert(A, 0, np.ones(A.shape[0]), axis=1)
b = train_data_array[:, -1]

test_data_array = test_data.to_numpy()
A_test = test_data_array[:, :-1]
A_test = np.insert(A_test, 0, np.ones(A_test.shape[0]), axis=1)
b_test = test_data_array[:, -1]


def bgd_method(A, b, epsilon, t=0.01):
    """ Batch Gradient Descent Method

    Args:
        A (mxn numpy array): input array holding m samples with n features
        b (mx1 numpy array): output 
        epsilon (float): tolerance level
        t (float): learning rate

    Returns:
        tree (dict): dictionary structure represented the decision tree

    """
    x = np.zeros(A.shape[1])

    diff = 1

    iter = 0
    fun_val = f(A, b, x)
    fun_history = fun_val

    grad = g(A, b, x)
    while (np.linalg.norm(diff) > epsilon):
        iter = iter+1

        # break out of while loop if diverging
        if np.linalg.norm(diff) > 1e20:
            break

        # define new point x = x + t d, d = - grad
        x_new = x-t*grad
        diff = x_new - x
        x = x_new

        fun_val = f(A, b, x)
        fun_history = np.vstack((fun_history, fun_val))
        grad = g(A, b, x)
        # if iter % 100 == 0:
        #print('iter_number = {}, tol = {:.4e}, fun_val = {:.4e}'.format(iter, np.linalg.norm(diff), fun_val))

    if np.linalg.norm(diff) > 1e20:
        iter = None
        print('Algorithm does not converge!')
    print('iter_number = {}, tol = {:.4e}, fun_val = {:.4e}'.format(
        iter, np.linalg.norm(diff), fun_val))
    return x, fun_history


def f(A, b, x):
    """ Calculates the loss funtion value
    """
    val = 0.0

    for i in range(A.shape[0]):
        val += (b[i] - np.dot(x, A[i, :]))**2

    return val / 2


def g(A, b, x):
    """ Calculates the gradient
    """
    grad = np.zeros(x.shape)

    for j in range(len(grad)):
        for i in range(A.shape[0]):
            grad[j] += (b[i] - np.dot(x, A[i, :]))*A[i, j]

    return - grad


def sgd_method(df, attrs, target_attr, epsilon, t=0.01):
    """ Stochastic Gradient Descent Method

    Args:
        A (mxn numpy array): input array holding m samples with n features
        b (mx1 numpy array): output 
        epsilon (float): tolerance level
        t (float): learning rate

    Returns:
        tree (dict): dictionary structure represented the decision tree

    """

    m, n = df.shape

    # convert to numpy array
    data = df.to_numpy()

    # Separate to Ax=b where A is input matrix, x is vector of weights, and b the vector of outputs
    A = data[:, :-1]
    A = np.insert(A, 0, np.ones(m), axis=1)
    b = data[:, -1]
    x = np.zeros(n)

    iter = 0
    max_iter = 100000
    cur_val = 100
    prev_val = np.inf
    history = cur_val

    while np.linalg.norm(prev_val-cur_val) > epsilon and iter < max_iter:
        iter = iter + 1
        prev_val = cur_val

        # shuffle indexes for sampling
        indexes = np.random.randint(m, size=m)

        for i in indexes:
            # define new point x = x + t d, d = - grad
            x = x + t*(b[i] - np.dot(x, A[i]))*A[i]

        cur_val = f(A, b, x)
        history = np.vstack((history, cur_val))

        #print('i = {}, tol = {:.4e}, fun_val = {:.4e}'.format(iter, np.linalg.norm(prev_val-cur_val), cur_val))

    print('i = {}, tol = {:.4e}, fun_val = {:.4e}'.format(
        iter, np.linalg.norm(prev_val-cur_val), cur_val))
    return x, history


def test_bgd():
    x1, fh1 = bgd_method(A, b, epsilon, t=1)
    x2, fh2 = bgd_method(A, b, epsilon, t=0.5)
    x3, fh3 = bgd_method(A, b, epsilon, t=0.25)
    x4, fh4 = bgd_method(A, b, epsilon, t=0.125)
    x5, fh5 = bgd_method(A, b, epsilon, t=0.0625)
    x6, fh6 = bgd_method(A, b, epsilon, t=0.03125)
    x7, fh7 = bgd_method(A, b, epsilon, t=0.015625)
    x8, fh8 = bgd_method(A, b, epsilon, t=0.0078125)
    x9, fh9 = bgd_method(A, b, epsilon, t=0.01)

    print(x9, fh9)

    plt.plot(range(len(fh9)), fh9)
    plt.title('Cost function vs Time')
    plt.xlabel('Iteration #')
    plt.ylabel('Cost Function Value')

    #plt.xlim([-100, 1000])

    plt.savefig('BGD_cost_vs_time.pdf')
    plt.show()

    print('Cost: {}'.format(f(A_test, b_test, x9)))


def test_sgd():
    x_b, h_b = sgd_method(test_data, features, output, epsilon, t=0.00001)

    plt.figure()
    plt.plot(range(len(h_b)), h_b)
    plt.title('Cost function vs Updates')
    plt.xlabel('Number of Updates')
    plt.ylabel('Cost Function Value')

    #plt.xlim([-100, 1000])
    plt.ylim([14, 24])

    plt.savefig('SGD_cost_vs_time.pdf')
    plt.show()

    print('Cost: {}'.format(f(A_test, b_test, x_b)))
