import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch import optim
import torch.nn as nn
import numpy as np
from amylib.utils.logtools import print_date_time

def system_cross_train(network : nn.Module, 
                       loader : DataLoader, 
                       epochs : int, 
                       lr : float | list, 
                       epoch_treshold : int, 
                       optimizer_type : str = 'ada', 
                       device : str = 'cpu', 
                       logging : int = 100):

    '''
    Performs training with 2 different optimizers (optimizer_type and SGD) 
    for a system of subnetworks

    args:

    network - model to be trained
    loader - data loader

    lr : float or list of floats - learning rate (or 1 learning rate for each optimizer)
    epoch_treshold - epoch after which the function switches optimizer to SGD
    logging : int - defines a number of pairs (epoch - system time) to print in console

    return:

    losses - mean loss of all sub-nets
    all_losses - array of loss curves of sub-nets
    grads - array of modules of gradient of sub-nets
    dirs - array of directions of gradient of sub-nets
    '''
    
    network.to(device)
    
    network.train()
    loss_function = nn.MSELoss(reduction = 'none')

    if type(lr) == float:
        lr0 = lr
        lr1 = lr
    else:
        lr0 = lr[0]
        lr1 = lr[1]
    
    if optimizer_type == 'ada':
        optimizer = optim.Adagrad(network.parameters(), lr = lr0)
    elif optimizer_type == 'sgd':
        optimizer = optim.SGD(network.parameters(), lr = lr0)
    elif optimizer_type == 'adam':
        optimizer = optim.Adam(network.parameters(), lr = lr0)
    else:
        raise ValueError('Incorrect optimizer type \" {} \"'.format(optimizer_type))
    
    batch_size = loader.batch_size
    batches = len(loader)
    
    losses = []
    all_losses = []

    grads = np.zeros((epochs, network.elements))
    dirs = np.zeros((epochs, network.elements))

    print('Training started ')
    print_date_time()

    switch = 1

    if logging > 0:
        logg_part = (epochs/logging)
        logg_flag = True
    
    for e in range(0, epochs):
        
        if logg_flag:
            if (e + 1)%logg_part == 0:
                print('Epoch ', e, ' ended')
                print_date_time()
                
        current_all_losses = torch.zeros(network.elements).to(device)
        current_loss = 0
        
        for i, (data, labels) in enumerate(loader):

            var_data = Variable(data)
            var_tgt = Variable(labels.repeat(1, network.elements))
            output = network.forward(var_data)
            loss = torch.sum(torch.mean(loss_function(output, var_tgt), dim = 0))
            current_all_losses += torch.mean((output - labels)**2, axis = 0)
            current_loss += loss.item()

            if e >= epoch_treshold and switch:
                optimizer = optim.SGD(network.parameters(), lr = lr1)
                switch = 0

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            grads[e] = network.get_module_grads()
            dirs[e] = network.get_grads_direction()

        losses.append(current_loss/batches)
        all_losses.append((current_all_losses/batches).cpu().detach().numpy())
            
    if logging > 0:
        print('')
        
    all_losses = np.array(all_losses).transpose()
    losses = np.array(losses)
            
    return losses, all_losses, grads, dirs


def system_test(network, x, target_function):
    
    network.eval()
    
    out = network.forward(torch.tensor(x.reshape(len(x), 1)))
    res = out.detach().numpy()
    true = target_function(x)
    
    return res, true