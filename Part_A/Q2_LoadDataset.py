import torch
import torchvision
import os
from torchvision import transforms
from torch.utils.data import random_split

'''Class containing the core functionality for loading and splitting the dataset.'''
class DatasetLoader:
    '''Constructor to set all the class parameters (core part).'''
    def __init__(self, root, batch_size):
        '''path of the dataset'''
        self.root = root
        '''batch size'''
        self.batch_size = batch_size
        '''transformation to apply on the dataset'''
        self.transform = transforms.Compose([
            transforms.Resize((112,112)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4747786223888397, 0.4644955098628998, 0.3964916169643402],
                                 std=[0.2389, 0.2289, 0.2422]),
        ])
        self.train_dataset, self.val_dataset, self.test_dataset = self.load_and_split_datasets()

    
    # def load_and_split_datasets(self):
    #     '''Loads and splits the original dataset.
        
    #     Returns:
    #         train_dataset: dataset for training
    #         val_dataset  : dataset for validation
    #         test_dataset : dataset for testing
    #     '''
    #     train_path = ''
    #     test_path = ''
    #     if "\\" in self.root:
    #         train_path = self.root + "/train"
    #         test_path = self.root + "/val"
    #     else:
    #         train_path = self.root + "\train"
    #         test_path = self.root + "\val"
    #     train_val_dataset = torchvision.datasets.ImageFolder(root=train_path, transform=self.transform)
    #     '''splitting into train and val'''
    #     train_size = int(0.8 * len(train_val_dataset))
    #     val_size = len(train_val_dataset) - train_size
    #     train_dataset, val_dataset = random_split(train_val_dataset, [train_size, val_size])
    #     test_dataset = torchvision.datasets.ImageFolder(root=test_path, transform=self.transform)
    #     return train_dataset, val_dataset, test_dataset
    
    def load_and_split_datasets(self):
        '''Loads and splits the original dataset.
        
        Returns:
            train_dataset: dataset for training
            val_dataset  : dataset for validation
            test_dataset : dataset for testing
        '''
        # Use os.path.join() to build paths reliably across platforms
        train_path = os.path.join(self.root, "train")
        test_path = os.path.join(self.root, "val")
        train_val_dataset = torchvision.datasets.ImageFolder(root=train_path, transform=self.transform)
        
        '''splitting into train and val'''
        train_size = int(0.8 * len(train_val_dataset))
        val_size = len(train_val_dataset) - train_size
        train_dataset, val_dataset = random_split(train_val_dataset, [train_size, val_size])
        
        test_dataset = torchvision.datasets.ImageFolder(root=test_path, transform=self.transform)
        return train_dataset, val_dataset, test_dataset
