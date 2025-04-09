import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, inputDepth, numOfFilters, sizeFilter, stride, padding, sizeDenseUnits, filterOrganization, activation, batchNormalization, dropoutProb):
        '''Constructor to set all the class parameters'''
        super(CNN, self).__init__()
        
        ''' Setting hyperparameters '''
        self.inputDepth = inputDepth
        self.outputDepth = numOfFilters
        self.stride = stride
        self.padding = padding
        self.neruonsInDenseLayer = sizeDenseUnits
        self.filterSize = sizeFilter
        self.filterOrganization = filterOrganization
        self.activation = activation
        self.batchNormalization = batchNormalization
        self.dropoutProb = dropoutProb
        
        ''' image dimensions - setting image size to 112 '''
        self.widhtOfImage = 112
        
        ''' first convolution-activation-maxpool block '''
        self.convLayer1 = nn.Conv2d(self.inputDepth, self.outputDepth, self.filterSize, self.stride, self.padding)
        self.bacthNormal1 = nn.BatchNorm2d(self.outputDepth)
        self.widhtOfImage = ((self.widhtOfImage - self.filterSize + 2*self.padding) // self.stride) + 1
        self.activationLayer1 = self.activationFunction(self.activation)
        self.maxPool1 = nn.MaxPool2d(self.filterSize, self.stride, self.padding)
        self.widhtOfImage = ((self.widhtOfImage - self.filterSize + 2*self.padding) // self.stride) + 1
        
        ''' Update for Block 2 '''
        self.inputDepth = self.outputDepth
        self.outputDepth = self.filterSizeCalculator(self.inputDepth, self.filterOrganization)
        
        ''' second convolution-activation-maxpool block '''
        self.convLayer2 = nn.Conv2d(self.inputDepth, self.outputDepth, self.filterSize, self.stride, self.padding)
        self.bacthNormal2 = nn.BatchNorm2d(self.outputDepth)
        self.widhtOfImage = ((self.widhtOfImage - self.filterSize + 2*self.padding) // self.stride) + 1
        self.activationLayer2 = self.activationFunction(self.activation)
        self.maxPool2 = nn.MaxPool2d(self.filterSize, self.stride, self.padding)
        self.widhtOfImage = ((self.widhtOfImage - self.filterSize + 2*self.padding) // self.stride) + 1
        
        ''' Update for Block 3 '''
        self.inputDepth = self.outputDepth
        self.outputDepth = self.filterSizeCalculator(self.inputDepth, self.filterOrganization)
        
        ''' third convolution-activation-maxpool block '''
        self.convLayer3 = nn.Conv2d(self.inputDepth, self.outputDepth, self.filterSize, self.stride, self.padding)
        self.bacthNormal3 = nn.BatchNorm2d(self.outputDepth)
        self.widhtOfImage = ((self.widhtOfImage - self.filterSize + 2*self.padding) // self.stride) + 1
        self.activationLayer3 = self.activationFunction(self.activation)
        self.maxPool3 = nn.MaxPool2d(self.filterSize, self.stride, self.padding)
        self.widhtOfImage = ((self.widhtOfImage - self.filterSize + 2*self.padding) // self.stride) + 1
        
        ''' Update for Block 4 '''
        self.inputDepth = self.outputDepth
        self.outputDepth = self.filterSizeCalculator(self.inputDepth, self.filterOrganization)
        
        ''' fourth convolution-activation-maxpool block '''
        self.convLayer4 = nn.Conv2d(self.inputDepth, self.outputDepth, self.filterSize, self.stride, self.padding)
        self.bacthNormal4 = nn.BatchNorm2d(self.outputDepth)
        self.widhtOfImage = ((self.widhtOfImage - self.filterSize + 2*self.padding) // self.stride) + 1
        self.activationLayer4 = self.activationFunction(self.activation)
        self.maxPool4 = nn.MaxPool2d(self.filterSize, self.stride, self.padding)
        self.widhtOfImage = ((self.widhtOfImage - self.filterSize + 2*self.padding) // self.stride) + 1
        
        ''' Update for Block 5 '''
        self.inputDepth = self.outputDepth
        self.outputDepth = self.filterSizeCalculator(self.inputDepth, self.filterOrganization)
        
        ''' fifth convolution-activation-maxpool block '''
        self.convLayer5 = nn.Conv2d(self.inputDepth, self.outputDepth, self.filterSize, self.stride, self.padding)
        self.bacthNormal5 = nn.BatchNorm2d(self.outputDepth)
        self.widhtOfImage = ((self.widhtOfImage - self.filterSize + 2*self.padding) // self.stride) + 1
        self.activationLayer5 = self.activationFunction(self.activation)
        self.maxPool5 = nn.MaxPool2d(self.filterSize, self.stride, self.padding)
        self.widhtOfImage = ((self.widhtOfImage - self.filterSize + 2*self.padding) // self.stride) + 1
        
        ''' post-convolution layers '''
        self.flatten = nn.Flatten() #flattening the output of the last maxpool layer
        self.dropout = nn.Dropout(self.dropoutProb)  #applying dropout
        self.fullyConnected = nn.Linear(self.widhtOfImage*self.widhtOfImage*self.outputDepth, self.neruonsInDenseLayer)  #defining the dense layer
        self.activationLayer6 = self.activationFunction(self.activation)  #applying activation function on the dense layer
        self.outputLayer = nn.Linear(self.neruonsInDenseLayer, 10)  #defining the output layer
