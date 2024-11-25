import numpy as np
import torch

def diag_mask(mh : int, mw : int, white_size : int) -> np.ndarray:

    '''
    Creates a matrix with size mh*mw of 0 and 1 

    args:
    
    mh : int - height of a mask
    mw : int - width of a mask

    white_size : int - number of 1 in each column

    return:

    mask : np.ndarray - a matrix with size mh*mw of 0 and 1
    '''
    
    mask = torch.zeros((mh, mw), requires_grad = False)
    
    h = 0
    for w in range(0, mw):
        for inh in range(0, white_size):
            mask[h * white_size + inh][w] = 1
        h +=1
        
    return mask