import numpy as np 


#-------------------------- helper functions ----------------------
# MAY NEED TO MOVE THEM IN A SEPERATE FILE 

def compute_mse(y,tx,w):
    """Calculate the loss using MSE

    Args:
        y: shape=(N,)
        tx: shape=(N,D)
        w: shape=(D,). The vector of model parameters.
        
        with D the features number 

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    # SEE personal not from series 2 for the calculations

    e = y - np.matmul(tx,w)
    N = y.shape[0]
    mse = (0.5*(1/N))*(e.T@e)  # factor 0.5 to stay consitant with lecture notes
    return mse

def compute_gradient(y, tx, w):
    """Computes the gradient at w.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,D)
        w: numpy array of shape=(D, ). The vector of model parameters.

    Returns:
        An numpy array of shape (D, ) (same shape as w), containing the gradient of the loss at w.
    """
   
    # SEE exo2 instruction (eq7) 
    N= y.shape[0]
    e = y - np.matmul(tx,w)
    grad = -(1/N)*(tx.T@e)
    return grad


def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient at w from a data sample batch of size B, where B < N, and their corresponding labels.

    Args:
        y: numpy array of shape=(B, )
        tx: numpy array of shape=(B,D)
        w: numpy array of shape=(D, ). The vector of model parameters.

    Returns:
        A numpy array of shape (D, ) (same shape as w), containing the stochastic gradient of the loss at w.
    """
    N= y.shape[0]
    e = y - np.matmul(tx,w)
    stoch_grad = -(1/N)*(tx.T@e)
    return stoch_grad


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.

    Example:

     Number of batches = 9

     Batch size = 7                              Remainder = 3
     v     v                                         v v
    |-------|-------|-------|-------|-------|-------|---|
        0       7       14      21      28      35   max batches = 6

    If shuffle is False, the returned batches are the ones started from the indexes:
    0, 7, 14, 21, 28, 35, 0, 7, 14

    If shuffle is True, the returned batches start in:
    7, 28, 14, 35, 14, 0, 21, 28, 7

    To prevent the remainder datapoints from ever being taken into account, each of the shuffled indexes is added a random amount
    8, 28, 16, 38, 14, 0, 22, 28, 9

    This way batches might overlap, but the returned batches are slightly more representative.

    Disclaimer: To keep this function simple, individual datapoints are not shuffled. For a more random result consider using a batch_size of 1.

    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)  # NUmber of data points.
    batch_size = min(data_size, batch_size)  # Limit the possible size of the batch.
    max_batches = int(
        data_size / batch_size
    )  # The maximum amount of non-overlapping batches that can be extracted from the data.
    remainder = (
        data_size - max_batches * batch_size
    )  # Points that would be excluded if no overlap is allowed.

    if shuffle:
        # Generate an array of indexes indicating the start of each batch
        idxs = np.random.randint(max_batches, size=num_batches) * batch_size
        if remainder != 0:
            # Add an random offset to the start of each batch to eventually consider the remainder points
            idxs += np.random.randint(remainder + 1, size=num_batches)
    else:
        # If no shuffle is done, the array of indexes is circular.
        idxs = np.array([i % max_batches for i in range(num_batches)]) * batch_size

    for start in idxs:
        start_index = start  # The first data point of the batch
        end_index = (
            start_index + batch_size
        )  # The first data point of the following batch
        yield y[start_index:end_index], tx[start_index:end_index]
#-----------------------end of helper functions ------------


def mean_squared_error_gd(y, tx, initial_w,max_iters, gamma):
   
    """The Gradient Descent (GD) algorithm using the MSE.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,D)
        initial_w: numpy array of shape=(D, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize

    Returns: A MODIFIER 
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of GD
        ws: a list of length max_iters containing the model parameters as numpy arrays of shape (D, ), for each iteration of GD
    """

    ws = initial_w
    losses = compute_mse(y, tx, ws) 
    for n_iter in range(max_iters):
        
        grad = compute_gradient(y, tx, ws)
       
        ws = ws - gamma*grad
        losses = compute_mse(y, tx, ws)
    return (ws,losses)




def mean_squared_error_sgd(y, tx, initial_w,max_iters, gamma):
    """The Stochastic Gradient Descent algorithm (SGD) using MSE with a batch size of 1 

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,D)
        initial_w: numpy array of shape=(D, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of SGD
        gamma: a scalar denoting the stepsize

    Returns: A MODIFIER 
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of SGD
        ws: a list of length max_iters containing the model parameters as numpy arrays of shape (2, ), for each iteration of SGD
    """

    batch_size = 1  #here we use a pure SGD (meaning we only sample 1 datapoint each iteration)
    # Define parameters to store w and loss
    ws = initial_w
    losses = compute_mse(y, tx, ws)


    for n_iter in range(max_iters):

        for y_batch,tx_batch in batch_iter(y, tx, batch_size): 
            stoch_grad = compute_stoch_gradient(y_batch,tx_batch,ws)
            
        ws = ws - gamma*stoch_grad
        losses = compute_mse(y, tx, ws)
      
    return (ws,losses)



def least_squares(y, tx): 
    0

def ridge_regression(y, tx, lambda_):
    0

def logistic_regression(y, tx, initial_w,max_iters, gamma):
    0

def reg_logistic_regression(y, tx, lambda_,initial_w, max_iters, gamma):
    0

