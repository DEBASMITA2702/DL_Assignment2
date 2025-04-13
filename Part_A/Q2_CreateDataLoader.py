import torch
from torch.utils.data import DataLoader
from Q2_LoadDataset import DatasetLoader

'''Class extending the core functionality to create data loaders.'''
class DatasetLoader_create(DatasetLoader):
    '''This class inherits from DatasetLoader and adds the data_loaders method.'''
    def data_loaders(self):
        '''Creates DataLoader objects for training, validation, and testing.
        
        Returns:
            train_loader : Data loader object of torch to be used for training
            val_loader : Data loader object of torch to be used for validation
            test_loader : Data loader object of torch to be used for testing
        '''
        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)
        return train_loader, val_loader, test_loader
