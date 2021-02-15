from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

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
    dW = np.zeros(W.shape) # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        loss_count = 0    # keep track of how many other classes contribute to the loss
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1 # note delta = 1
            if margin > 0:
                loss += margin
                # if didn't meet the desired margin, means the incorrect class contributes to the loss
                dW[:,j] += X[i]    # take all rows, but only needs the jth column in each eow
                loss_count += 1

        # gradient on the correct class, depends on the loss count of other classes
        dW[:,y[i]] += (-1) * loss_count * X[i]
        
    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    dW += 2* reg * W    # take the gradient of the regularization as well

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather than first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    # done in the above code

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    return loss, dW



def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape) # initialize the gradient as zero
    
    num_classes = W.shape[1]
    num_train = X.shape[0]
    
    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    scores = X.dot(W)    # (N,D) dot (D,C) gives (N,C)
    
    # select the yth column for each row, get (N,)
    correct_class_score = scores[range(num_train),y]  
    
    # reshape it into (N,1))
    correct_class_score = correct_class_score.reshape(num_train, 1)    
    
    # (N,C) - (N,1), we still have (N,C)
    margin = scores - correct_class_score + 1    
    
    # the correct class don't contribute loss
    margin[range(num_train),y] = 0    
    
    # construct loss function
    loss = np.sum(np.fmax(margin, 0)) / num_train 
    
    # add regularization
    loss += reg * np.sum(W * W)

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    # inside margin, count the number of positive terms in each row, store it as a mask
    mask = np.zeros(margin.shape) 
    mask[margin > 0] = 1
    positive_count = np.sum(mask, axis=1)    # this is (N,)
    
    # add gradients on the correct class, will time X[i] later together
    mask[range(num_train), y] += (-1) * positive_count 
    dW = (X.T).dot(mask) / num_train    # this is (D,N) dot (N,C) = (D,C)
 
    dW += 2* reg * W
    
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
