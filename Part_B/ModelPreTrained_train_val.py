from ModelPreTrained_Base import FineTuningModelBase

"""Extending FineTuningModelBase by adding the training and validation methods."""
class FineTuningModelTrainVal(FineTuningModelBase):
    """Inherits from FineTuningModelBase and implements training and validation methods."""
    
    def training_step(self, batch, batch_idx):
        """
        Parameters:
            batch: A batch of training data (input tensor and labels).
            batch_idx: Batch index.
        Returns:
            loss: Loss value calculated after forward propagation and loss computation.
        Function:
            Performs a training step by processing the input, computing the loss, updating training metrics, and logging the loss.
        """
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        
        self.training_accuracy(y_hat, y)
        self.log("training_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        return loss
    
    def on_train_epoch_end(self):
        """
        Parameters:
            None
        Returns:
            training accuracy after resetting the training metric.
        Function:
            Logs the training accuracy at the end of an epoch and resets the training accuracy metric.
        """
        self.log('training_accuracy', self.training_accuracy.compute(), prog_bar=True, logger=True)
        return self.training_accuracy.reset()
    
    def validation_step(self, batch, batch_idx):
        """
        Parameters:
            batch: A batch of validation data (input tensor and labels).
            batch_idx: Batch index.
        Returns:
            loss: Loss value calculated on the validation batch.
        Function:
            Performs a validation step by processing the input, computing the loss, updating validation metrics, and logging the loss.
        """
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)

        self.validation_accuracy(y_hat, y)
        self.log("validation_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        return loss
    
    def on_validation_epoch_end(self):
        """
        Parameters:
            None
        Returns:
            validation accuracy after resetting the validation metric.
        Function:
            Logs the validation accuracy at the end of an epoch and resets the validation accuracy metric.
        """
        self.log("validation_accuracy", self.validation_accuracy.compute(), prog_bar=True, logger=True)
        return self.validation_accuracy.reset()
