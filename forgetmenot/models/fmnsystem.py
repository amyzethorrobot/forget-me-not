import torch
import numpy as np
import torch.nn as nn
from ..utils.mask import diag_mask

class fmnSystem(nn.Module):
    
    def __init__(self, 
                 elements : int, 
                 hidden_size : int, 
                 device : str = 'cpu', 
                 bias : bool = True):
        
        super(fmnSystem, self).__init__()

        '''
        System of networks class constructor

        args:

        elemets : int - number of subnetworks
        hidden_size : int - size of the hidden layer
        device - 
        bias : bool - bias in neurons

        
        self.hidden_weights - weght matrix between hidden and output layer
        self.mask - mask to cut connections between hidden and ouput neurons 
        of different subnets
        
        '''

        # some variables

        self.bias = bias

        if bias:
            self.hidden_bias = nn.Parameter(torch.zeros(elements))
        else:
            self.hidden_bias = None

        self.device = device
        self.elements = elements
        self.hidden_size = hidden_size

        # layers
        
        self.full = nn.Sequential(
            nn.Linear(1, elements * hidden_size, bias = self.bias), 
            nn.Sigmoid()
        )

        torch.nn.init.normal_(self.full[0].weight, 0, 1)
        

        self.hidden_weights = nn.Parameter((torch.normal(0, 
                                                         1/hidden_size, 
                                                         (elements * hidden_size, elements)
                                                        ) * 
                                           diag_mask(elements * hidden_size, 
                                                      elements, 
                                                      hidden_size)).transpose(0, 1))
        
        self.mask = nn.Parameter(diag_mask(elements * hidden_size, 
                                            elements, 
                                            hidden_size).transpose(0, 1), 
                                 requires_grad = False)
        
        self.out_nonlin = nn.Sigmoid()

        self.stored_grads = torch.zeros(self.elements, self.hidden_size * 2)
        
        #self.double()
        
    def forward(self, x):

        x = self.full(x)
        pre_out = torch.nn.functional.linear(x, (self.hidden_weights * self.mask), self.hidden_bias)
        out = self.out_nonlin(pre_out)
        return out

    def get_module_grads(self):

        '''
        Returns module of gradient vector at current step
        '''

        module_grads = np.zeros(self.elements)

        with torch.no_grad():
            for i in range(0, self.elements):

                fp = i * self.hidden_size
                sp = fp + self.hidden_size
                
                module_grads[i] += torch.sum(self.full[0].weight.grad[fp:sp]**2)
                module_grads[i] += torch.sum(self.hidden_weights.grad[i][fp:sp]**2)

            module_grads = np.sqrt(module_grads)
            
        return module_grads

    def get_grads_direction(self):

        '''
        Returns direction of the gradient vector at current step
        '''

        current_grads = torch.zeros(self.elements, self.hidden_size * 2)
        directions = np.zeros(self.elements)

        for i in range(0, self.elements):

            fp = i * self.hidden_size
            sp = fp + self.hidden_size
            first_grad = torch.flatten(self.full[0].weight.grad[fp:sp])
            second_grad = self.hidden_weights.grad[i][fp:sp]

            current_grads[i][0:self.hidden_size] = first_grad
            current_grads[i][self.hidden_size:self.hidden_size * 2] = second_grad

        for i in range(0, self.elements):
            direction = torch.dot(self.stored_grads[i], current_grads[i])
            directions[i] = direction

        self.stored_grads = current_grads

        return directions