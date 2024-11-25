import numpy as np

def hann_function(x, width = 10):
    window = lambda x, N: 0.5 * (1 - 
                                  np.cos(2 * np.pi * 
                                         x / (N - 1)))
    return window(x, width + 1)