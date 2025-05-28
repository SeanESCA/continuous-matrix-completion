from __future__ import annotations
import numpy as np
import torch
from time import time


def syntheticA(n: int) -> torch.tensor:
    """
    Produces a synthetic rank-2 n-by-n matrix
    """
    X = torch.vstack((torch.arange(n)/n, (1 - torch.arange(n)/n)**2)).T

    return X @ X.T

def initial(A: torch.tensor, r: float) -> list:
    """
    Randomly initialises the matrix factors X and Y.
    """
    X0 = torch.rand(A.size(0), r)*1e-3
    Y0 = torch.rand(A.size(1), r)*1e-3
    X0.requires_grad_()
    Y0.requires_grad_()

    return [X0, Y0]

def bernoulli(A: torch.tensor, p: float) -> torch.tensor:
    """
    Produces a set of index pairs I = {(i1,i2)} where
    i1 = 0,...,(n1)-1, i2 = 0,...,(n2)-1, and each pair 
    is sampled with probability p.
    """
    I = []
    for i1 in range(A.size(0)):
        for i2 in range(A.size(1)):
            if np.random.rand() < p:
                I.append([i1,i2])
    np.random.shuffle(I)

    return torch.tensor(I)

def make_mask(n1: int, n2: int, I: torch.tensor) -> torch.tensor:
    '''
    Produces a mask based on the index set I.
    '''
    mask = torch.zeros((n1,n2), dtype=torch.int8)
    for i in I:
        mask[i[0], i[1]] = 1
    return mask

def completion_err(X: torch.tensor, Y: torch.tensor, A: torch.tensor) -> float:
    '''
    Computes the mean completion error between the full matrix A
    and the solution X(Y.T).
    '''
    return torch.norm(X@Y.T - A)/A.norm()

# Loss Functions

mse_loss = torch.nn.MSELoss()
asd_loss = lambda X, Y: 0.5*((X - Y)**2).sum()

def batch_loss( X: torch.tensor, 
                Y: torch.tensor, 
                A: torch.tensor, 
                I: torch.tensor,
                loss: function = mse_loss):
    '''
    Evaluates the loss function over index set I.
    '''
    return loss(torch.einsum(
        'in, in -> i',
        torch.index_select(X, 0, I[:,0]), 
        torch.index_select(Y, 0, I[:,1])
    ), A[I[:,0], I[:,1]])

# Algorithms for standard matrix completion.

def optimise(optimiser, X, Y, A, I, B=1, dk=100, K=10000, k_test=1, k_true=int(1e7), q=0.99):
    
    # Compute the training mini-batch size.
    batch_size = int((I.size(0) * 0.9) // B)

    # Compute the starting index of the test set.
    test_set_start = batch_size * B

    train_loss_list = []
    test_loss_list = []
    true_err_list = []
    timestamps = []
    best_params = []
    best_test_loss = 1e6

    for k in range(K):
        train_loss = batch_loss(X, Y, A, I[((k % B) * batch_size):((k % B + 1) * batch_size)])
        train_loss_list.append(train_loss.item())
        timestamps.append(time())

        # Compute the test loss.
        if (k % k_test == 0):
            test_loss = batch_loss(X, Y, A, I[test_set_start:])
            test_loss_list.append(test_loss.item())
            
            # Save best parameters.
            if test_loss < best_test_loss:
                best_test_loss = test_loss.item()
                best_params = [X.detach().clone(), Y.detach().clone()]

            # Early stopping condition.
            if (k >= dk) and (test_loss > (q * test_loss_list[(k - dk)//k_test])):
                break

        # Compute the mean completion error.
        if (k % k_true):
            true_err = completion_err(X, Y, A)
            true_err_list.append(true_err.item())
        
        train_loss.backward()
        optimiser.step()
        optimiser.zero_grad()

    print(f"Best mean completion error: {completion_err(best_params[0], best_params[1], A)}")

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

def asd_1(U, W, Y, I, B=1, dk=100, q=0.99, K=1000, lr=4):

    U_ind, W_ind = I.T
    Y_ind = U_ind*Y.shape[1] + W_ind
    batch_size = int((len(Y_ind)*0.9)//B)
    test_set_start = batch_size*B

    train_loss_list = []
    test_loss_list = []
    true_err_list = []
    timestamps = []
    lr_list = []
    best_U = None
    best_W = None
    best_test_loss = 1e6
    best_iter = 0

    optimiserU = torch.optim.SGD([U], lr=lr)
    optimiserW = torch.optim.SGD([W], lr=lr)

    for k in range(K):
        timestamps.append(time())
        i = (k % B)*batch_size
        j = (k % B + 1)*batch_size
        
        # Optimise U for fixed W.
        train_loss = batch_loss(
            U, W, Y, U_ind[i:j], W_ind[i:j], Y_ind[i:j], asd_loss
        )
        test_loss = batch_loss(U, W, Y, 
                               U_ind[test_set_start:], W_ind[test_set_start:], Y_ind[test_set_start:])
        # true_err = completion_err(U, W, Y)

        train_loss_list.append(train_loss.item())
        test_loss_list.append(test_loss.item())
        # true_err_list.append(true_err.item())

        # Save best parameters.
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_U = U.detach().clone()
            best_W = W.detach().clone()
            best_iter = k

        # Early stopping.
        if (k >= dk) and (test_loss > q * test_loss_list[k - dk]):
            break
        
        train_loss.backward()
        lrU = (U.grad.detach().norm()/torch.einsum(
            'in, in -> i',
            torch.index_select(U.grad.detach(), 0, U_ind[i:j]), 
            torch.index_select(W.detach(), 0, W_ind[i:j])
        ).norm())**2
        optimiserU.param_groups[0]["lr"] = lrU 
        optimiserU.step()
        optimiserU.zero_grad()
        optimiserW.zero_grad()

        # Optimise W for fixed U.
        train_loss = batch_loss(
            U, W, Y, U_ind[i:j], W_ind[i:j], Y_ind[i:j], asd_loss
        )

        train_loss.backward()
        lrW = (W.grad.detach().norm()/torch.einsum(
            'in, in -> i',
            torch.index_select(U.detach(), 0, U_ind[i:j]), 
            torch.index_select(W.grad.detach(), 0, W_ind[i:j])
        ).norm())**2
        optimiserW.param_groups[0]["lr"] = lrW
        optimiserW.step()
        optimiserW.zero_grad()
        optimiserU.zero_grad()
        lr_list.append((lrU, lrW))
    print(f"{k}, {best_iter}, {completion_err(best_U, best_W, Y)}")

    return {
        "params": [U, W],
        "train_loss_list": train_loss_list,
        "test_loss_list": test_loss_list,
        "true_err_list": true_err_list,
        "timestamps": timestamps,
        "best_params": [best_U, best_W],
        "best_iter": best_iter,
        "lr_list": lr_list
    }

def asd_2(U, W, Y, I, B=1, dk=100, q=0.99, K=1000, lr=4):

    U_ind, W_ind = I.T
    Y_ind = U_ind*Y.shape[1] + W_ind
    batch_size = int((len(Y_ind)*0.9)//B)
    test_set_start = batch_size*B

    train_loss_list = []
    test_loss_list = []
    true_err_list = []
    timestamps = []
    best_U = None
    best_W = None
    best_test_loss = 1e6
    best_iter = 0
    
    lr_list_1 = []
    lr_list_2 = []

    optimiserU = torch.optim.SGD([U], lr=lr)
    optimiserW = torch.optim.SGD([W], lr=lr)

    for k in range(K):
        timestamps.append(time())
        i = (k % B)*batch_size
        j = (k % B + 1)*batch_size
        
        # Optimise U for fixed W.
        train_loss = batch_loss(
            U, W, Y, U_ind[i:j], W_ind[i:j], Y_ind[i:j], asd_loss
        )
        test_loss = batch_loss(U, W, Y, 
                               U_ind[test_set_start:], W_ind[test_set_start:], Y_ind[test_set_start:])
        # true_err = completion_err(U, W, Y)

        train_loss_list.append(train_loss.item())
        test_loss_list.append(test_loss.item())
        # true_err_list.append(true_err.item())

        # Save best parameters.
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_U = U.detach().clone()
            best_W = W.detach().clone()
            best_iter = k

        # Early stopping.
        if (k >= dk) and (test_loss > q * test_loss_list[k - dk]):
            break
        
        train_loss.backward()
        lrU = (U.grad.detach().norm()/torch.einsum(
            'in, in -> i',
            torch.index_select(U.grad.detach(), 0, U_ind[i:j]), 
            torch.index_select(W.detach(), 0, W_ind[i:j])
        ).norm())**2
        optimiserU.param_groups[0]["lr"] = lrU 
        optimiserU.step()
        optimiserU.zero_grad()
        optimiserW.zero_grad()

        # Optimise W for fixed U.
        # i = ((k + 1) % B)*batch_size
        # j = ((k + 1) % B + 1)*batch_size
        i = ((k + B//2) % B)*batch_size
        j = ((k + B//2) % B + 1)*batch_size
        train_loss = batch_loss(
            U, W, Y, U_ind[i:j], W_ind[i:j], Y_ind[i:j], asd_loss
        )

        train_loss.backward()
        lrW = (W.grad.detach().norm()/torch.einsum(
            'in, in -> i',
            torch.index_select(U.detach(), 0, U_ind[i:j]), 
            torch.index_select(W.grad.detach(), 0, W_ind[i:j])
        ).norm())**2
        optimiserW.param_groups[0]["lr"] = lrW
        optimiserW.step()
        optimiserW.zero_grad()
        optimiserU.zero_grad()
        lr_list_1.append((lrU, lrW))
        lr_list_2.append((optimiserU.param_groups[0]["lr"], optimiserW.param_groups[0]["lr"]))

    print(f"{k}, {best_iter}, {completion_err(best_U, best_W, Y)}")

    return {
        "params": [U, W],
        "train_loss_list": train_loss_list,
        "test_loss_list": test_loss_list,
        "true_err_list": true_err_list,
        "timestamps": timestamps,
        "best_params": [best_U, best_W],
        "best_iter": best_iter,
        "lr_list_1": lr_list_1,
        "lr_list_2": lr_list_2,
    }

def raw_asd(U, W, Y, I, B=1, dk=100, q=0.99, K=1000):

    U_ind, W_ind = I.T
    Y_ind = U_ind*Y.shape[1] + W_ind
    batch_size = int((len(Y_ind)*0.9)//B)
    test_set_start = batch_size*B

    test_loss_list = []
    true_err_list = []
    lr_list = []
    best_U = None
    best_W = None
    best_test_loss = 1e6
    best_iter = 0

    for k in range(K):
        i = (k % B)*batch_size
        j = (k % B + 1)*batch_size
        mask = make_mask(Y.shape[0],Y.shape[1],I[i:j])
        
        # Optimise U for fixed W.
        test_loss = batch_loss(U, W, Y, 
                               U_ind[test_set_start:], W_ind[test_set_start:], Y_ind[test_set_start:])
        true_err = completion_err(U, W, Y)

        test_loss_list.append(test_loss.item())
        true_err_list.append(true_err.item())

        # Save best parameters.
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_U = U.detach().clone()
            best_W = W.detach().clone()
            best_iter = k

        # Early stopping.
        if (k >= dk) and (test_loss > q * test_loss_list[k - dk]):
            break
        
        U_grad = - (Y*mask - (U.detach()@W.detach().T)*mask) @ W.detach()
        lrU = (U_grad.norm()/((U_grad@W.T)*mask).norm())**2
        U = U - lrU*U_grad

        # Optimise W for fixed U.
        
        W_grad = - U.detach().T @ (Y*mask - (U.detach()@W.detach().T)*mask)
        lrW = (W_grad.norm()/((U@W_grad)*mask).norm())**2
        W = W - W_grad.T*lrW
        lr_list.append((lrU, lrW))

    print(f"{k}, {best_iter}, {completion_err(best_U, best_W, Y)}")

    return {
        "params": [U, W],
        "test_loss_list": test_loss_list,
        "true_err_list": true_err_list,
        "best_params": [best_U, best_W],
        "best_iter": best_iter,
        "lr_list": lr_list
    }

# Continuous matrix completion.
def syntheticf(x):
    """
    Synthetic function yielding a rank-2 matrix
    x must be a 2D tensor with shape (M,2)
    """
    return x[:,0]*x[:,1] + ((1-x[:,0])**2)*((1-x[:,1])**2)

def optimise_f(optimiser, params, F, X,Y, B=1, dk=100, q=0.99, K=10000):
    
    U_ind, W_ind = X.T
    batch_size = int((len(F)*0.9)//B)
    test_set_start = batch_size*B

    train_loss_list = []
    test_loss_list = []
    true_err_list = []
    timestamps = []
    best_U = None
    best_W = None
    best_test_loss = 1e6
    best_iter = 0

    for k in range(K):
        timestamps.append(time())
        i = (k % B)*batch_size
        j = (k % B + 1)*batch_size
        train_loss = new_loss(params[0], params[1], F[i:j], U_ind[i:j], W_ind[i:j])
        test_loss = new_loss(params[0], params[1], F[test_set_start:], 
                             U_ind[test_set_start:], W_ind[test_set_start:])
        true_err = completion_err(params[0], params[1], Y)

        train_loss_list.append(train_loss.item())
        test_loss_list.append(test_loss.item())
        true_err_list.append(true_err.item())
        
        # Save best parameters.
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_U = params[0].detach().clone()
            best_W = params[1].detach().clone()
            best_iter = k

        # Early stopping.
        if (k >= dk) and (train_loss > q * train_loss_list[k - dk]):
            break
        
        train_loss.backward()
        optimiser.step()
        optimiser.zero_grad()

    print(f"{k}, {best_iter}, {completion_err(best_U, best_W, Y)}")

    return {
        "final_params": params,
        "train_loss_list": train_loss_list,
        "test_loss_list": test_loss_list,
        "true_err_list": true_err_list,
        "timestamps": timestamps,
        "best_params": [best_U, best_W],
        "best_iter": best_iter
    }

def asd_f(U, W, F, Y, I, B=1, dk=100, q=0.99, K=1000, lr=4):

    U_ind, W_ind = I.T
    batch_size = int((len(F)*0.9)//B)
    test_set_start = batch_size*B

    train_loss_list = []
    test_loss_list = []
    true_err_list = []
    timestamps = []
    best_U = None
    best_W = None
    best_test_loss = 1e6
    best_iter = 0

    optimiserU = torch.optim.SGD([U], lr=lr)
    optimiserW = torch.optim.SGD([W], lr=lr)

    for k in range(K):
        timestamps.append(time())
        i = (k % B)*batch_size
        j = (k % B + 1)*batch_size
        
        # Optimise U for fixed W.
        train_loss = new_loss(
            U, W, F[i:j], U_ind[i:j], W_ind[i:j], asd_loss
        )
        test_loss = new_loss(U, W, F[test_set_start:], 
                               U_ind[test_set_start:], W_ind[test_set_start:])
        true_err = completion_err(U, W, Y)

        # train_loss_list.append(train_loss.item())
        test_loss_list.append(test_loss.item())
        true_err_list.append(true_err.item())

        # Save best parameters.
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_U = U.detach().clone()
            best_W = W.detach().clone()
            best_iter = k

        # Early stopping.
        if (k >= dk) and (test_loss > q * test_loss_list[k - dk]):
            break
        
        train_loss.backward()
        lrU = (U.grad.detach().norm()/linterp2d(U.grad.detach(),W,U_ind[i:j],W_ind[i:j]).norm())**2
        optimiserU.param_groups[0]["lr"] = lrU 
        optimiserU.step()
        optimiserU.zero_grad()
        optimiserW.zero_grad()

        # Optimise W for fixed U.
        train_loss = new_loss(
            U, W, F[i:j], U_ind[i:j], W_ind[i:j], asd_loss
        )

        train_loss.backward()
        lrW = (W.grad.detach().norm()/linterp2d(U,W.grad.detach(),U_ind[i:j],W_ind[i:j]).norm())**2
        optimiserW.param_groups[0]["lr"] = lrW
        optimiserW.step()
        optimiserW.zero_grad()
        optimiserU.zero_grad()

    print(f"{k}, {best_iter}, {completion_err(best_U, best_W, Y)}")

    return {
        "params": [U, W],
        "train_loss_list": train_loss_list,
        "test_loss_list": test_loss_list,
        "true_err_list": true_err_list,
        "timestamps": timestamps,
        "best_params": [best_U, best_W],
        "best_iter": best_iter
    }

# Interpolation.
def linterp1d(x:torch.tensor, points:torch.tensor, h=1):
    '''
    1D piecewise linear interpolation using torch.

    Inputs:
    - x (torch.tensor): 1D tensor. Every element of x must be in the interval [0, len(points) - 1].
    - h (int)         : Step size.
    '''
        
    # Calculate the coordinates of the nearest data points to x.
    x1 = (x // h).int()
    x2 = x1 + h

    # Modify indexing for elements of x that correspond to the last data point.
    max_ind = points.shape[0] - 1
    x1[x1 >= max_ind] = max_ind - 1 
    x2[x2 > max_ind] = max_ind
    return ((x2 - x).reshape((-1,1))*torch.index_select(points,0,x1) + 
            (x - x1).reshape((-1,1))*torch.index_select(points,0,x2))/h

def linterp2d(U: torch.tensor, W: torch.tensor, U_ind:torch.tensor, W_ind: torch.tensor, h=1):
    uu = linterp1d(U_ind, U, h)
    ww = linterp1d(W_ind, W, h)
    return torch.einsum('in, in -> i', uu, ww)

def new_loss( U: torch.tensor, 
                W: torch.tensor, 
                F: torch.tensor, 
                U_ind,
                W_ind,
                loss=mse_loss):
    return loss(linterp2d(U,W,U_ind,W_ind), F)