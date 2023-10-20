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
    e = y - np.matmul(tx,w) 
    N = y.shape[0]
    mse = (1/2*N)*(e.T@e) 
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
   
   
    N= y.shape[0]
    e = y - np.matmul(tx,w)
    grad = -(1/N)*(tx.T@e)
    return grad


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
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        
        grad = compute_gradient(y, tx, ws[n_iter])
        loss = compute_mse(y, tx, ws[n_iter])
       
        w = ws[n_iter] - gamma*grad
        

        # store w and loss
        ws.append(w)
        losses.append(loss)

    return losses(0), ws(0)
