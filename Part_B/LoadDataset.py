import torch
import torchvision
from torchvision import transforms
import os
from torch.utils.data import random_split

'''class to load the dataset'''
class DatasetLoader:
    '''constructor to set all the class parameters'''
    def __init__(self, root, batch_size):
        '''path of the dataset'''
        self.root = root
        '''batch size'''
        self.batch_size = batch_size
        '''transformation to apply on the dataset'''
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4747786223888397, 0.4644955098628998, 0.3964916169643402],std=[0.2389, 0.2289, 0.2422]),
        ])
        self.train_dataset, self.val_dataset, self.test_dataset = self.load_and_split_datasets()

    def load_and_split_datasets(self):
        '''
        Parameters:
            None
        Function:
            Loads and splits the original dataset
        Returns:
            train_dataset: dataset for training
            val_dataset: dataset for validation
            test_dataset: dataset for testing
        '''
        train_path = ''
        test_path = ''
        train_path = os.path.join(self.root, "train")
        test_path = os.path.join(self.root, "val")
        train_val_dataset = torchvision.datasets.ImageFolder(root=train_path, transform=self.transform)
        
        '''splitting into train and val'''
        train_size = int(0.8 * len(train_val_dataset))
        val_size = len(train_val_dataset) - train_size
        train_dataset, val_dataset = random_split(train_val_dataset, [train_size, val_size])
        
        test_dataset = torchvision.datasets.ImageFolder(root=test_path, transform=self.transform)
        return train_dataset, val_dataset, test_dataset