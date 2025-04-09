def forward(self, x):
    '''Forward propagation with all 5 blocks
    Parameters:
        x : object to apply forward propagation on
    Returns :
        x : the same object after application of forward propagation
    Function:
        Applies forward propagation
    '''
    # Block 1
    if self.batchNormalization == "Yes":
        x = self.maxPool1(self.activationLayer1(self.bacthNormal1(self.convLayer1(x))))
    else:
        x = self.maxPool1(self.activationLayer1(self.convLayer1(x)))
    
    # Block 2
    if self.batchNormalization == "Yes":
        x = self.maxPool2(self.activationLayer2(self.bacthNormal2(self.convLayer2(x))))
    else:
        x = self.maxPool2(self.activationLayer2(self.convLayer2(x)))
    
    # Block 3
    if self.batchNormalization == "Yes":
        x = self.maxPool3(self.activationLayer3(self.bacthNormal3(self.convLayer3(x))))
    else:
        x = self.maxPool3(self.activationLayer3(self.convLayer3(x)))
    
    # Block 4
    if self.batchNormalization == "Yes":
        x = self.maxPool4(self.activationLayer4(self.bacthNormal4(self.convLayer4(x))))
    else:
        x = self.maxPool4(self.activationLayer4(self.convLayer4(x)))
    
    # Block 5
    if self.batchNormalization == "Yes":
        x = self.maxPool5(self.activationLayer5(self.bacthNormal5(self.convLayer5(x))))
    else:
        x = self.maxPool5(self.activationLayer5(self.convLayer5(x)))
    
    ''' Final layers '''
    x = self.flatten(x)
    x = self.dropout(x)
    x = self.fullyConnected(x)
    x = self.activationLayer6(x)
    x = self.outputLayer(x)
    return x
