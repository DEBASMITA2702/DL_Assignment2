import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import lightning as pl
import torch.optim as optim

'''Class to run PyTorch Lightning (core functionality)'''
class FastRunningCore(pl.LightningModule):

    '''Constructor to set all the class parameters'''
    def __init__(self, model):
        super(FastRunningCore, self).__init__()
        '''Setting the model defined by the CNN class'''
        self.model = model

        '''Setting the loss type'''
        self.criterion = nn.CrossEntropyLoss()

        '''Metrics to store the accuracies'''
        self.training_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=10)
        self.validation_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=10)
    
    def configure_optimizers(self):
        '''
        Parameters:
            None
        Returns :
            optimizer : the optimizer object that will be applied on the network
        Function:
            Creates an object of the optimizer and returns it
        '''
        optimizer = optim.SGD(self.model.parameters(), lr=0.0001, momentum=0.8, nesterov=True)
        return optimizer
    
    def training_step(self, batch, batch_idx):
        '''
        Parameters:
            batch : images in batches
            batch_idx : batch id of the corresponding batch
        Returns :
            loss : loss obtained after backpropagation
        Function:
            Does the training step
        '''
        x, y = batch
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)
        predicted = torch.argmax(y_hat, dim=1)
        self.training_accuracy(predicted, y)
        self.log("training_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        '''
        Parameters:
            batch : images in batches
            batch_idx : batch id of the corresponding batch
        Returns :
            loss : loss obtained after backpropagation
        Function:
            Does the validation step
        '''
        x, y = batch
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)
        predicted = torch.argmax(y_hat, dim=1)
        self.validation_accuracy(predicted, y)
        self.log("validation_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss
