import torch
import numpy as np

def dataset_function(x : np.ndarray, 
                     target_function : callable, 
                     device : str = 'cpu') -> torch.utils.data.dataset.TensorDataset:

    '''
    Creates dataset for regression task

    arguments:

    x : np.ndarray - array of arguments
    target_function : callable - callable function which returns y = target_function(x) values
    device : str - which device will be used to store dataset

    returns:

    ds : torch.utils.data.dataset.TensorDataset - dataset
    '''
    
    labels = target_function(x)
    
    x_r = x.reshape(len(x), 1)
    l_r = labels.reshape(len(labels), 1)
    
    x_t = torch.tensor(x_r).to(device)
    l_t = torch.tensor(l_r).to(device)
    
    ds = torch.utils.data.TensorDataset(x_t, l_t)
    
    return ds

def loader_function(dataset: torch.utils.data.dataset.TensorDataset, 
                    batch_size : int = 1, 
                    shuffle : bool = True) -> torch.utils.data.dataloader.DataLoader:

    ''' 
    Creates DataLoader for dataset for regression task

    dataset : torch.utils.data.dataset.TensorDataset - dataset
    batch_size : int - size of mini-batch
    shuffle : bool - shuffling data
    '''
    
    loader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle = shuffle)
    return loader