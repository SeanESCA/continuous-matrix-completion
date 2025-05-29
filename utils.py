from __future__ import annotations
import numpy as np
import torch
from time import time


def syntheticA(n: int) -> torch.tensor:
    """
    Produces a synthetic rank-2 square matrix.

    Inputs:
    - n (int): Size of the matrix.

    Outputs:
    - A torch tensor of shape (n, n).
    """
    X = torch.vstack((torch.arange(n)/n, (1 - torch.arange(n)/n)**2)).T

    return X @ X.T

def initial(A: torch.tensor, r: float) -> list:
    """
    Randomly initialises the matrix factors X and Y of the solution X(Y.T).

    Inputs:
    - A (torch.tensor): Matrix of shape (n1, n2).
    - r (int): Rank of the solution .

    Outputs:
    - X0 (torch.tensor): Matrix of shape (n1, r).
    - Y0 (torch.tensor): Matrix of shape (n2, r).
    """
    X0 = torch.rand(A.size(0), r)*1e-3
    Y0 = torch.rand(A.size(1), r)*1e-3
    X0.requires_grad_()
    Y0.requires_grad_()

    return [X0, Y0]

def bernoulli(A: torch.tensor, p: float) -> torch.tensor:
    """
    Produces a set of index pairs I = {(i1,i2)} where i1 = 0,...,(n1)-1, 
    i2 = 0,...,(n2)-1, and each pair is sampled with probability p.

    Inputs:
    - A (torch.tensor): Matrix of shape (n1, n2).
    - p (float): Sampling probability.

    Outputs:
    - I (torch.tensor): Index matrix. Each row contains an index pair.
    """
    I = []
    for i1 in range(A.size(0)):
        for i2 in range(A.size(1)):
            if np.random.rand() < p:
                I.append([i1,i2])
    np.random.shuffle(I)

    return torch.tensor(I)

def make_mask(A: torch.tensor, I: torch.tensor) -> torch.tensor:
    '''
    Produces a mask based on the index set I.

    Inputs:
    - A (torch.tensor): Matrix of shape (n1, n2).
    - I (torch.tensor): Index matrix.

    Outputs:
    - A torch tensor of shape (n1, n2).
    '''
    n1, n2 = A.shape
    mask = torch.zeros((n1,n2), dtype=torch.int8)
    for i in I:
        mask[i[0], i[1]] = 1
    return mask

def completion_err(X: torch.tensor, Y: torch.tensor, A: torch.tensor) -> torch.tensor:
    '''
    Computes the mean completion error between the full matrix A
    and the solution X(Y.T).

    Inputs:
    - X (torch.tensor): Matrix of shape (n1, r).
    - Y (torch.tensor): Matrix of shape (n2, r).
    - A (torch.tensor): Matrix of shape (n1, n2).

    Outputs:
    - A torch tensor that contains a single element.
    '''
    return (torch.norm(X@Y.T - A)/A.norm()).item()

def syntheticF(I: torch.tensor):
    """
    Samples the synthetic rank-2 matrix with jittered positions.
    Inputs:
    - I (torch.tensor): Exact, jittered index matrix.

    Outputs:
    - A torch.tensor containing the entries
    """
    return I[:,0] * I[:,1] + ((1 - I[:,0])**2)*((1 - I[:,1])**2)

# Interpolation.

def linterp1d(x:torch.tensor, points:torch.tensor):
    '''
    1-D piecewise linear interpolation for vectors.

    Inputs:
    - x (torch.tensor): every element of x must be in the interval [0, len(points) - 1].
    - points (torch.tensor): 
    '''
        
    # Calculate the coordinates of the nearest data points to x.
    x1 = x.int()
    x2 = x1 + 1

    # Modify indexing for elements of x that correspond to the last data point.
    max_ind = points.shape[0] - 1
    x1[x1 >= max_ind] = max_ind - 1 
    x2[x2 > max_ind] = max_ind
    return ((x2 - x).reshape((-1,1))*torch.index_select(points,0,x1) + 
            (x - x1).reshape((-1,1))*torch.index_select(points,0,x2))/h

def linterp2d(X: torch.tensor, Y: torch.tensor, I:torch.tensor) -> torch.tensor:
    '''
    Bilinear interpolation on a low-rank matrix A = X(Y.T) with rank r.

    Inputs:
    - X (torch.tensor): Matrix of shape (n1, r).
    - Y (torch.tensor): Matrix of shape (n2, r).
    - I (torch.tensor): Index matrix of shape (m, 2).

    Outputs:
    - A torch.tensor of length m.
    '''
    xx: torch.tensor = linterp1d(I[:,0], X)
    yy: torch.tensor = linterp1d(I[:,1], Y)
    return torch.einsum('in, in -> i', xx, yy)

# Loss Functions

mse_loss = torch.nn.MSELoss()
asd_loss = lambda X, Y: 0.5*((X - Y)**2).sum()

def mse_loss_std(X: torch.tensor, Y: torch.tensor, A: torch.tensor, I: torch.tensor) -> torch.tensor:
    '''
    Evaluates the MSE between A and X(Y.T) over the index set I.
    
    Inputs:
    - X (torch.tensor): Matrix of shape (n1, r).
    - Y (torch.tensor): Matrix of shape (n2, r).
    - A (torch.tensor): Matrix of shape (n1, n2).
    - I (torch.tensor): Index matrix.
    - loss (function): Loss function.

    Outputs:
    - A torch.tensor that contains a float. 
    '''
    return mse_loss(torch.einsum(
        'in, in -> i',
        torch.index_select(X, 0, I[:,0]), 
        torch.index_select(Y, 0, I[:,1])
    ), A[I[:,0], I[:,1]])

def asd_loss_std(X: torch.tensor, Y: torch.tensor, A: torch.tensor, I: torch.tensor) -> torch.tensor:
    '''
    Evaluates the ASD loss between A and X(Y.T) over the index set I.
    
    Inputs:
    - X (torch.tensor): Matrix of shape (n1, r).
    - Y (torch.tensor): Matrix of shape (n2, r).
    - A (torch.tensor): Matrix of shape (n1, n2).
    - I (torch.tensor): Index matrix.
    - loss (function): Loss function.

    Outputs:
    - A torch.tensor that contains a float. 
    '''
    return asd_loss(torch.einsum(
        'in, in -> i',
        torch.index_select(X, 0, I[:,0]), 
        torch.index_select(Y, 0, I[:,1])
    ), A[I[:,0], I[:,1]])

def mse_loss_cont(X: torch.tensor, Y: torch.tensor, F: torch.tensor, I: torch.tensor) -> torch.tensor:
    '''
    Evaluates the MSE loss between A and X(Y.T) over the continuous index set I.
    
    Inputs:
    - X (torch.tensor): Matrix of shape (n1, r).
    - Y (torch.tensor): Matrix of shape (n2, r).
    - F (torch.tensor): Sample of known entries.
    - I (torch.tensor): Index matrix.
    - loss (function): Loss function.

    Outputs:
    - A torch.tensor that contains a float. 
    '''
    return mse_loss(linterp2d(X,Y,I), F)

def asd_loss_cont(X: torch.tensor, Y: torch.tensor, F: torch.tensor, I: torch.tensor) -> torch.tensor:
    '''
    Evaluates the ASD loss between A and X(Y.T) over the continuous index set I.
    
    Inputs:
    - X (torch.tensor): Matrix of shape (n1, r).
    - Y (torch.tensor): Matrix of shape (n2, r).
    - F (torch.tensor): Sample of known entries.
    - I (torch.tensor): Index matrix.
    - loss (function): Loss function.

    Outputs:
    - A torch.tensor that contains a float. 
    '''
    return asd_loss(linterp2d(X,Y,I), F)

def zero_fn(*args): return 0

# Algorithms for standard matrix completion.

def optimise(X: torch.tensor, Y: torch.tensor, A: torch.tensor, I: torch.tensor, 
             algo: str = "sgd", lr: float = 1, loss: function = mse_loss_std, 
             true_err_fn: function = zero_fn, B: int = 1, dk: int = 100, 
             K=10000, k_test=1, k_true=int(1e7), q=0.99) -> dict:
    '''
    Optimise X and Y such that A = X(Y.T) using SGD or Adam.
    
    Inputs:
    - X (torch.tensor): Matrix of shape (n1, r).
    - Y (torch.tensor): Matrix of shape (n2, r).
    - A (torch.tensor): Matrix of shape (n1, n2).
    - I (torch.tensor): Index matrix.
    - algo (str): If not "adam", then SGD will be used.
    - lr (float): Learning rate.
    - loss (function): Function that takes X, Y, A, and I as arguments.
    - true_err_fn (function): Function that takes X and Y as arguments.
    - B (int): Number of mini-batches.
    - dk (int): Iteration gap for comparing the test loss.
    - K (int): Maximum number of iterations.
    - k_test (int): Iteration gap for computing the test loss.
    - k_true (int): Iteration gap for computing the mean completion error.
    - q (float): Relative change in the test loss.

    Outputs:
    - train_loss_list (list): Training loss (float) for every iteration.
    - test_loss_list (list): Test loss (float) for every (k_test)-th iteration.
    - true_err_list (list): Mean completion error (float) for every (k_true)-th iteration.
    - timestamps (list): Time after computing the training loss for every iteration.
    - best_params (list): Factors [X, Y] with the lowest test loss.
    - B (int): Number of mini-batches.
    - dk (int): Iteration gap for comparing the test loss.
    - K (int): Maximum number of iterations.
    - k_test (int): Iteration gap for computing the test loss.
    - k_true (int): Iteration gap for computing the mean completion error.
    - q (float): Relative change in the test loss.
    - lr (float): Learning rate.
    '''
    # Initialise the optimiser. 
    if (algo == "adam"):
        optimiser = torch.optim.Adam([X, Y], lr=lr)
    else:
        optimiser = torch.optim.SGD([X, Y], lr=lr)

    # Compute the size of each training mini-batch.
    batch_size = int((I.size(0) * 0.9) // B)

    # Compute the starting index of the test set.
    test_set_start: int = batch_size * B

    train_loss_list = []
    test_loss_list = []
    true_err_list = []
    timestamps = []
    best_params = []
    best_test_loss = 1e6

    for k in range(K):
        # Compute the training loss
        train_loss = loss(X, Y, A, I[((k % B) * batch_size):((k % B + 1) * batch_size)])
        train_loss_list.append(train_loss.item())
        timestamps.append(time())

        # Compute the test loss.
        if (k % k_test == 0):
            test_loss = loss(X, Y, A, I[test_set_start:])
            test_loss_list.append(test_loss.item())
            
            # Update the best parameters.
            if test_loss < best_test_loss:
                best_test_loss = test_loss.item()
                best_params = [X.detach().clone(), Y.detach().clone()]

            # Early stopping condition.
            if (k >= dk) and (test_loss > (q * test_loss_list[(k - dk)//k_test])):
                break

        # Compute the mean completion error.
        if (k % k_true == 0):
            true_err = true_err_fn(X, Y)
            true_err_list.append(true_err)
        
        train_loss.backward()
        optimiser.step()
        optimiser.zero_grad()

    print(f"Best mean completion error: {true_err_fn(best_params[0], best_params[1])}")

    return {
        "train_loss_list": train_loss_list,
        "test_loss_list": test_loss_list,
        "true_err_list": true_err_list,
        "timestamps": timestamps,
        "best_params": best_params,
        "final_params": [X, Y],
        "k_test": k_test,
        "k_true": k_true,
        "B": B,
        "dk": dk,
        "K": K,
        "q": q,
        "lr": optimiser.param_groups[0]["lr"]
    }

def asd(X: torch.tensor, Y: torch.tensor, A: torch.tensor, I: torch.tensor, 
        train_loss_fn: function = asd_loss_std, test_loss_fn: function = mse_loss_std, 
        true_err_fn: function = zero_fn, B: int = 1, db: int = 0, dk: int = 100, 
        K=10000, k_test=1, k_true=int(1e7), q=0.99) -> dict:
    '''
    Optimise X and Y such that A = X(Y.T) using mini-batch ASD.
    
    Inputs:
    - X (torch.tensor): Matrix of shape (n1, r).
    - Y (torch.tensor): Matrix of shape (n2, r).
    - A (torch.tensor): Matrix of shape (n1, n2).
    - I (torch.tensor): Index matrix.
    - train_loss_fn (function): Function that takes X, Y, A, and I as arguments.
    - test_loss_fn (function): Function that takes X, Y, A, and I as arguments.
    - true_err_fn (function): Function that takes X and Y as arguments.
    - B (int): Number of mini-batches.
    - db (int): Mini-batch counter gap between X and Y.
    - dk (int): Iteration gap for comparing the test loss.
    - K (int): Maximum number of iterations.
    - k_test (int): Iteration gap for computing the test loss.
    - k_true (int): Iteration gap for computing the mean completion error.
    - q (float): Relative change in the test loss.

    Outputs:
    - train_loss_list (list): Training loss (float) for every iteration.
    - test_loss_list (list): Test loss (float) for every (k_test)-th iteration.
    - true_err_list (list): Mean completion error (float) for every (k_true)-th iteration.
    - timestamps (list): Time after computing the training loss for every iteration.
    - best_params (list): Factors [X, Y] with the lowest test loss.
    - B (int): Number of mini-batches.
    - dk (int): Iteration gap for comparing the test loss.
    - K (int): Maximum number of iterations.
    - k_test (int): Iteration gap for computing the test loss.
    - k_true (int): Iteration gap for computing the mean completion error.
    - q (float): Relative change in the test loss.
    - lrX_list (list): Learning rates for optimising X in each iteration.
    - lrY_list (list): Learning rates for optimising Y in each iteration.
    '''
    # Initialise the optimisers. 
    optimiserX = torch.optim.SGD([X])
    optimiserY = torch.optim.SGD([Y])

    # Compute the size of each training mini-batch.
    batch_size = int((I.size(0) * 0.9) // B)

    # Compute the starting index of the test set.
    test_set_start: int = batch_size * B

    train_loss_list = []
    test_loss_list = []
    true_err_list = []
    timestamps = []
    best_params = []
    lrX_list = []
    lrY_list = []
    best_test_loss = 1e6

    for k in range(K):
        # Compute the indexes for the start and end of the mini-batch
        i: int = (k % B) * batch_size
        j: int = (k % B + 1) * batch_size

        # Compute the training loss
        train_loss = train_loss_fn(X, Y, A, I[i:j])
        train_loss_list.append(train_loss.item())
        timestamps.append(time())

        # Compute the test loss.
        if (k % k_test == 0):
            test_loss = test_loss_fn(X, Y, A, I[test_set_start:])
            test_loss_list.append(test_loss.item())
            
            # Update the best parameters.
            if test_loss < best_test_loss:
                best_test_loss = test_loss.item()
                best_params = [X.detach().clone(), Y.detach().clone()]

            # Early stopping condition.
            if (k >= dk) and (test_loss > (q * test_loss_list[(k - dk)//k_test])):
                break

        # Compute the mean completion error.
        if (k % k_true == 0):
            true_err = true_err_fn(X, Y)
            true_err_list.append(true_err)
        
        train_loss.backward()

        # Adaptive learning rate for X.
        lrX = ((X.grad.norm()/torch.einsum(
            'in, in -> i',
            torch.index_select(X.grad, 0, I[i:j, 0]), 
            torch.index_select(Y, 0, I[i:j, 1])
        ).norm())**2).item()
        optimiserX.param_groups[0]["lr"] = lrX

        # Optimise X.
        optimiserX.step()
        optimiserX.zero_grad()
        optimiserY.zero_grad()

        # Compute the training loss.
        i: int = ((k + db) % B) * batch_size
        j: int = ((k + db) % B + 1) * batch_size
        train_loss = train_loss_fn(X, Y, A, I[i:j])
        train_loss.backward()

        # Adaptive learning rate for Y.
        lrY = ((Y.grad.norm()/torch.einsum(
            'in, in -> i',
            torch.index_select(X, 0, I[i:j, 0]), 
            torch.index_select(Y.grad, 0, I[i:j, 1])
        ).norm())**2).item()
        optimiserY.param_groups[0]["lr"] = lrY

        # Optimise Y.
        optimiserY.step()
        optimiserX.zero_grad()
        optimiserY.zero_grad()

        lrX_list.append(lrX)
        lrY_list.append(lrY)

    print(f"Best mean completion error: {true_err_fn(best_params[0], best_params[1])}")

    return {
        "train_loss_list": train_loss_list,
        "test_loss_list": test_loss_list,
        "true_err_list": true_err_list,
        "timestamps": timestamps,
        "best_params": best_params,
        "final_params": [X, Y],
        "k_test": k_test,
        "k_true": k_true,
        "B": B,
        "dk": dk,
        "K": K,
        "q": q,
        "lrX_list": lrX_list,
        "lrY_list": lrY_list
    }

def raw_asd(X: torch.tensor, Y: torch.tensor, A: torch.tensor, I: torch.tensor, 
        B: int = 1, db: int = 0, dk: int = 100, K=10000, k_test=1, 
        k_true=int(1e7), q=0.99) -> dict:
    '''
    Optimise X and Y such that A = X(Y.T) using mini-batch ASD. This version
    calculates the gradients and learning rates explicitly without any shortcuts, 
    which makes it incredibly slow for large matrices.
    
    Inputs:
    - X (torch.tensor): Matrix of shape (n1, r).
    - Y (torch.tensor): Matrix of shape (n2, r).
    - A (torch.tensor): Matrix of shape (n1, n2).
    - I (torch.tensor): Index matrix.
    - B (int): Number of mini-batches.
    - db (int): Mini-batch counter gap between X and Y.
    - dk (int): Iteration gap for comparing the test loss.
    - K (int): Maximum number of iterations.
    - k_test (int): Iteration gap for computing the test loss.
    - k_true (int): Iteration gap for computing the mean completion error.
    - q (float): Relative change in the test loss.

    Outputs:
    - test_loss_list (list): Test loss (float) for every (k_test)-th iteration.
    - true_err_list (list): Mean completion error (float) for every (k_true)-th iteration.
    - best_params (list): Factors [X, Y] with the lowest test loss.
    - B (int): Number of mini-batches.
    - dk (int): Iteration gap for comparing the test loss.
    - K (int): Maximum number of iterations.
    - k_test (int): Iteration gap for computing the test loss.
    - k_true (int): Iteration gap for computing the mean completion error.
    - q (float): Relative change in the test loss.
    - lrX_list (list): Learning rates for optimising X in each iteration.
    - lrY_list (list): Learning rates for optimising Y in each iteration.
    '''
    # Compute the size of each training mini-batch.
    batch_size = int((I.size(0) * 0.9) // B)

    # Compute the starting index of the test set.
    test_set_start: int = batch_size * B

    test_loss_list = []
    true_err_list = []
    best_params = []
    lrX_list = []
    lrY_list = []
    best_test_loss = 1e6

    for k in range(K):
        i: int = (k % B) * batch_size
        j: int = (k % B + 1) * batch_size
        mask = make_mask(A, I[i:j])
        
        # Compute the test loss.
        if (k % k_test == 0):
            test_loss = mse_loss_std(X, Y, A, I[test_set_start:])
            test_loss_list.append(test_loss.item())
            
            # Update the best parameters.
            if test_loss < best_test_loss:
                best_test_loss = test_loss.item()
                best_params = [X.detach().clone(), Y.detach().clone()]

            # Early stopping condition.
            if (k >= dk) and (test_loss > (q * test_loss_list[(k - dk)//k_test])):
                break

        # Compute the mean completion error.
        if (k % k_true == 0):
            true_err = completion_err(X, Y, A)
            true_err_list.append(true_err.item())
        
        X_grad = - (Y * mask - (X @ Y.T) * mask) @ Y
        lrX = (X_grad.norm() / ((X_grad @ Y.T) * mask).norm())**2
        X = X - lrX * X_grad

        # Optimise W for fixed U.
        i: int = ((k + db) % B) * batch_size
        j: int = ((k + db) % B + 1) * batch_size
        Y_grad = - X.T @ (Y * mask - (X @ Y.T) * mask)
        lrY = (Y_grad.norm() / ((X @ Y_grad) * mask).norm())**2
        Y = Y - Y_grad.T * lrY

        lrX_list.append(lrX)
        lrY_list.append(lrY)

    print(f"Best mean completion error: {completion_err(best_params[0], best_params[1], A)}")

    return {
        "test_loss_list": test_loss_list,
        "true_err_list": true_err_list,
        "best_params": best_params,
        "final_params": [X, Y],
        "k_test": k_test,
        "k_true": k_true,
        "B": B,
        "dk": dk,
        "K": K,
        "q": q,
        "lrX_list": lrX_list,
        "lrY_list": lrY_list
    }
