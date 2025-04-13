from Q2_LightningModel_Base import FastRunningCore
import torch

'''Class to run PyTorch Lightning with additional epoch-end logging methods.'''
class FastRunning(FastRunningCore):

    def on_train_epoch_end(self):
        '''
        Parameters:
            None
        Returns :
            training accuracy after resetting the metric to 0
        Function:
            Logs training accuracy after end of epoch
        '''
        self.log('training_accuracy', self.training_accuracy.compute(), prog_bar=True, logger=True)
        return self.training_accuracy.reset()
    
    def on_validation_epoch_end(self):
        '''
        Parameters:
            None
        Returns :
            validation accuracy after resetting the metric to 0
        Function:
            Logs validation accuracy after end of epoch
        '''
        self.log("validation_accuracy", self.validation_accuracy.compute(), prog_bar=True, logger=True)
        return self.validation_accuracy.reset()
