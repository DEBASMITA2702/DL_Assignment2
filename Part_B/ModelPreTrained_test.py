import torch
from ModelPreTrained_train_val import FineTuningModelTrainVal

"""Extending FineTuningModelTrainVal by adding the test step method"""
class FineTuningModel(FineTuningModelTrainVal):
    """Inherits from FineTuningModelTrainVal and implements the test_step method"""
    
    def test_step(self, batch, batch_idx):
        """
        Parameters:
            batch: A batch of test data (input tensor and labels).
            batch_idx: Batch index.
        Returns:
            accuracy: Test accuracy for the processed batch.
        Function:
            Performs a test step by processing the input, determining the predicted class,
            calculating the batch accuracy, and accumulating the test accuracy.
        """
        x, y = batch
        y_hat = self(x)
        predicted = torch.argmax(y_hat, dim=1)
        correct_points = (predicted == y).sum().item()
        total_points = len(y)
        accuracy = correct_points / total_points
        self.test_accuracy += accuracy
        return accuracy
