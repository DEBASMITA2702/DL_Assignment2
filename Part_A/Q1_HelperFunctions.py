import torch.nn as nn

def filterSizeCalculator(self, filterSize, filterOrganization):
    '''Filter size update logic
    Parameters:
        filterSize : current size of the filter
        filterOrganization : what type of organization to apply of the filter size
    Returns :
        filterSize : size of filter after applying the update rule
    Function:
        Applies filter organization
    '''
    if filterOrganization == "same":
        return filterSize
    if filterOrganization == "half" and filterSize > 1:
        return filterSize // 2
    if filterOrganization == "double" and filterSize <= 512:
        return filterSize * 2
    return filterSize

def activationFunction(self, activation):
    '''Activation function selector
    Parameters:
        activation : what type of activation function is to be applied
    Returns :
        object of the activation function
    Function:
        Creates and returns an object of the activation function
    '''
    if activation == "ReLU":
        return nn.ReLU()
    if activation == "GELU":
        return nn.GELU()
    if activation == "SiLU":
        return nn.SiLU()
    if activation == "Mish":
        return nn.Mish()
