import torch
import torchvision
import torch.nn as nn
import lightning as pl
import re
import torchmetrics
import torch.optim as optim

"""Base class to perform fine tuning on a pre-trained model.
This class initializes the pre-trained GoogLeNet model, freezes layers, sets up the feature extractor and attaches a new output layer to match the number of classes."""
class FineTuningModelBase(pl.LightningModule):
    """Constructor to set all the class parameters and initialize the pre-trained model."""
    def __init__(self, numClasses, numOfFreezedLayers, learning_rate, aux_logits=True):
        super(FineTuningModelBase, self).__init__()
        
        """Loads the GoogLeNet model pre-trained on ImageNet."""
        self.model = torchvision.models.googlenet(pretrained=True)
        self.numOfFreezedLayers = numOfFreezedLayers
        self.learning_rate = learning_rate

        """Freezes the layers based on the specified number of layers to freeze."""
        for n, p in self.model.named_parameters():
            match = re.search(r'\d+', n.split('.')[0])
            if match and int(match.group()) < self.numOfFreezedLayers:
                p.requires_grad = False
        
        """Extracts the layers excluding the final fully-connected layer as a feature extractor."""
        numOfLayers = list(self.model.children())[:-1]
        self.feature_extractor = nn.Sequential(*numOfLayers)
        self.feature_extractor.eval()

        """Attaches a new linear layer to match the output dimensions to the number of classes in the iNaturalist dataset."""
        inFeatures = self.model.fc.in_features
        self.outputLayer = nn.Linear(inFeatures, numClasses)

        """The cross-entropy loss function."""
        self.criterion = nn.CrossEntropyLoss()

        """Initializes metrics for training and validation accuracies and a variable for test accuracy."""
        self.training_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=numClasses)
        self.validation_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=numClasses)
        self.test_accuracy = 0
        
    def forward(self, x):
        """
        Parameters:
            x: Input tensor for forward propagation.
        Returns:
            x: Output tensor after applying the forward propagation.
        Function:
            Flattens the features extracted by the feature extractor and applies the output layer.
        """
        flattened = self.feature_extractor(x).flatten(1)
        x = self.outputLayer(flattened)
        return x
    
    def configure_optimizers(self):
        """
        Parameters:
            None
        Returns:
            optimizer: Optimizer object for training the network.
        Function:
            Creates and returns an Adam optimizer using the model parameters and the specified learning rate.
        """
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return optimizer
