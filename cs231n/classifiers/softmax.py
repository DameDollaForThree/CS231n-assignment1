from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    num_class = W.shape[1]    # this is C
    num_train = X.shape[0]    # this is N

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    scores = X.dot(W)    # we get (N,C)
    for i in range(num_train):
        score = scores[i] - np.max(scores[i])    # this is (C,)  # subtract the max to avoid numerical overflow
        
        exp_each_score = np.exp(score)    # this is still (C,)
        exp_score_sum = np.sum( exp_each_score )
        
        # compute the gradient descent
        for j in range(num_class):
            # gradient for both incorrect classes and correct classes, require some calculus calculations
            dW[:,j] += (exp_each_score[j] / exp_score_sum) * X[i]
           
        # gradient added on the correct class, require some calculus calculations, 
        #(notice that dW[:,y[i]] has been computed twice)
        dW[:,y[i]] -= X[i]
        
        
#         # Another way to calculate the gradient: completely separate incorrect and correct classes
#         for j in range(num_class):
#             # calculate the correct class separately
#             if j == y[i]:
#                 continue
#             dW[:,j] += (exp_each_score[j] / exp_score_sum) * X[i]
#         dW[:,y[i]] += (-1) * (exp_score_sum - exp_each_score[y[i]]) / exp_score_sum * X[i]
            
        # compute the loss  
        loss += (-1) * np.log( exp_each_score[y[i]] / exp_score_sum )
    
    loss /= num_train
    loss += reg * np.sum(W * W)
    
    dW /= num_train
    dW += 2*reg*W
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    num_class = W.shape[1]    # this is C
    num_train = X.shape[0]    # this is N

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    scores = X.dot(W)      # we get (N,C)
    scores = scores - np.reshape(np.max(scores, axis=1), (num_train, 1))    # avoid overflow
    
    exp_scores = np.exp(scores)    # still (N,C)
    exp_score_sum = np.sum(exp_scores, axis=1)    # this is (N,)
    
    loss = (-1) * np.sum( np.log( exp_scores[range(num_train), y] / exp_score_sum ) )
    
    loss /= num_train
    loss += reg * np.sum(W * W)
    
    softmax_score = exp_scores / exp_score_sum.reshape(num_train,1)   
    softmax_score[np.arange(num_train), y] -= 1
    dW = X.T.dot(softmax_score)    
    
    dW /= num_train
    dW += 2*reg*W
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
