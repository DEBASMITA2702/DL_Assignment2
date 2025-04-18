import wandb
import warnings
import argparse
import Q1_ForwardPass
import Q1_HelperFunctions
import Q1_CNNmodel
import Q2_CreateDataLoader             
import torch
import torch.nn as nn
from torchvision import transforms
from Q2_LightningModel_EpochLogs import FastRunning
from pytorch_lightning.loggers import WandbLogger
import lightning as pl
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

warnings.filterwarnings("ignore")

# Attach the forward method and helper functions to the CNN class:
Q1_CNNmodel.CNN.forward = Q1_ForwardPass.forward
Q1_CNNmodel.CNN.activationFunction = Q1_HelperFunctions.activationFunction
Q1_CNNmodel.CNN.filterSizeCalculator = Q1_HelperFunctions.filterSizeCalculator

'''login to wandb to generate plot'''
wandb.login()

'''setting the device to gpu if it is available'''
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

'''class to support the train_parta.py execution'''
class Train:
    def runTrain(project_name,root,epochs,batch_size,numOfFilters,sizeFilter,sizeDenseUnits,batchNormalization,dataAugmentation,dropoutProb,activation,filterOrganization,test):
        '''
        Parameters:
            project_name : naem of the wandb project
            root : absolute path of the dataset
            epochs : number of epochs to run
            batch_size : size of batch to divide the dataset into
            numOfFilters : number of filters in the input layer
            sizeFilter : size(dimension) of each filter
            sizeDenseUnits : Number of neurons in the dense layer
            bacthNormalization : boolean value indicating wheteher to apply batch normalization or not
            dataAugmentation : boolean value indicating wheteher to apply data augmentation or not
            dropoutProb : probability of dropout
            activation : activation fucntion that is to be applied
            filterOrganization : organization of the filters across the layers
            test : boolean value indicating wheteher to do testing or not
        Returns :
            None
        Function:
            Supports the execution of train_parta.py
        '''

        '''get the data loaders'''
        dataLoader = Q2_CreateDataLoader.DatasetLoader_create(root=root,batch_size=batch_size)
        trainLoader, valLoader, testLoader = dataLoader.data_loaders()

        '''if needed then apply data augmentation'''
        if dataAugmentation=="Yes":
            transform = transforms.Compose([
                transforms.Resize((112,112)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.2,contrast=0.2,saturation=0.2,hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4747786223888397,0.4644955098628998,0.3964916169643402],std=[0.2389, 0.2289, 0.2422]),
            ])
            trainLoader.dataset.transform = transform
            valLoader.dataset.transform = transform
            testLoader.dataset.transform = transform

        '''set wandb run name'''
        run = "EP_{}_FIL_{}_FILSIZE_{}_FCSIZE_{}_FILORG_{}_AC_{}_DRP_{}_BS_{}".format(epochs, numOfFilters, sizeFilter, sizeDenseUnits, filterOrganization, activation, dropoutProb, batch_size)
        wandb.run.name = run

        '''define and apply early stopping'''
        early_stop_callback = EarlyStopping(
            monitor="validation_accuracy",
            min_delta=0.01,
            patience=3,
            verbose=False,
            mode="max"
        )

        '''create and object of the CNN class that will have the model defined in it'''
        model = Q1_CNNmodel.CNN(
            inputDepth=3,
            numOfFilters=numOfFilters,
            sizeFilter=sizeFilter,
            stride=1,
            padding=2,
            sizeDenseUnits=sizeDenseUnits,
            filterOrganization=filterOrganization,
            activation=activation,
            batchNormalization=batchNormalization,
            dropoutProb=dropoutProb
        )

        '''define an object of the lightning class and pass the CNN class model into it to run it'''
        lightningModel = FastRunning(model)

        '''set a wandb logger'''
        wandb_logger = WandbLogger(project=project_name, log_model='all')

        '''define a trainer object to run the lightning object'''
        trainer = pl.Trainer(max_epochs=epochs, logger=wandb_logger)
        if device != torch.device('cpu'):
            trainer = pl.Trainer(max_epochs=epochs, devices=-1, logger=wandb_logger)

        '''training and validation step'''
        trainer.fit(lightningModel, trainLoader, valLoader)

        '''if needed then run test and print the accuracy'''
        if test == 1:
            correct = 0
            total = 0

            for image, label in testLoader:
                with torch.no_grad():
                    y_hat = lightningModel.model(image)
                    _, predicted = torch.max(y_hat, 1)
                    total += label.size(0)
                    correct += (predicted == label).sum().item()

            print("Test Accuracy : ", correct/total)

def arguments():
    '''
      Parameters:
        None
      Returns :
        A parser object
      Function:
        Does command line argument parsing and returns the arguments passed
    '''
    commandLineArgument = argparse.ArgumentParser(description='Model Parameters')
    commandLineArgument.add_argument('-wp','--wandb_project', help="Project name used to track experiments in Weights & Biases dashboard")
    commandLineArgument.add_argument('-we','--wandb_entity', help="Wandb Entity used to track experiments in the Weights & Biases dashboard")
    commandLineArgument.add_argument('-r','--root', help="Absolute path of the dataset")
    commandLineArgument.add_argument('-e','--epochs', type=int, help="Number of epochs to train neural network")
    commandLineArgument.add_argument('-b','--batch', type=int, help="Batch size to divide the dataset")
    commandLineArgument.add_argument('-f','--filter', type=int, help="Number of filters in the first convolution layer")
    commandLineArgument.add_argument('-fs','--filter_size', type=int, help="Dimension of the filters")
    commandLineArgument.add_argument('-n','--neurons', type=int, help="Number of neurons in the fully connected layer")
    commandLineArgument.add_argument('-bn','--batch_normal', help="choices: ['Yes','No']")
    commandLineArgument.add_argument('-da','--data_augment', help="choices: ['Yes','No']")
    commandLineArgument.add_argument('-d','--dropout', type=float, help="Percentage of dropout in the network")
    commandLineArgument.add_argument('-a','--activation', help="Activation function in the activation layers")
    commandLineArgument.add_argument('-fo','--filter_org', help="Organization of the filters across the layers")
    commandLineArgument.add_argument('-t','--test', type=int, help="choices: [0,1]")

    return commandLineArgument.parse_args()

'''main driver function'''
def main():
    '''default values of each of the hyperparameter. it is set according to the config of my best model :- EP_5_FIL_32_FILSIZE_5_FCSIZE_64_FILORG_same_AC_Mish_DRP_0.4_BS_32 '''
    project_name = 'Debasmita-DA6401-Assignment-2'
    entity_name = 'cs24m015-indian-institute-of-technology-madras'
    epochs = 5
    batch_size = 32
    numOfFilters = 32
    sizeFilter = 5
    sizeDenseUnits = 64
    batchNormalization = "Yes"
    dataAugmentation = "No"
    dropoutProb = 0.4
    activation = "Mish"
    filterOrganization = "same"
    test = 0
    root = '/kaggle/input/inaturalist-12k/nature_12K_dataset/inaturalist_12K'

    '''call to argument function to get the arguments'''
    argumentsPassed = arguments()

    '''checking if a particular argument is passed through command line or not and updating the values accordingly'''
    if argumentsPassed.wandb_project is not None:
        project_name = argumentsPassed.wandb_project
    if argumentsPassed.wandb_entity is not None:
        entity_name = argumentsPassed.wandb_entity
    if argumentsPassed.root is not None:
        root = argumentsPassed.root
    if argumentsPassed.epochs is not None:
        epochs = argumentsPassed.epochs
    if argumentsPassed.batch is not None:
        batch_size = argumentsPassed.batch
    if argumentsPassed.filter is not None:
        numOfFilters = argumentsPassed.filter
    if argumentsPassed.filter_size is not None:
        sizeFilter = argumentsPassed.filter_size
    if argumentsPassed.neurons is not None:
        sizeDenseUnits = argumentsPassed.neurons
    if argumentsPassed.batch_normal is not None:
        batchNormalization = argumentsPassed.batch_normal
    if argumentsPassed.data_augment is not None:
        dataAugmentation = argumentsPassed.data_augment
    if argumentsPassed.dropout is not None:
        dropoutProb = argumentsPassed.dropout
    if argumentsPassed.activation is not None:
        activation = argumentsPassed.activation
    if argumentsPassed.filter_org is not None:
        filterOrganization = argumentsPassed.filter_org
    if argumentsPassed.test is not None:
        test = argumentsPassed.test

    '''initializing to the project'''
    wandb.init(project=project_name, entity=entity_name)

    '''calling the functions with the parameters'''
    run = "EP_{}_FIL_{}_FILSIZE_{}_FCSIZE_{}_FILORG_{}_AC_{}_DRP_{}".format(epochs, numOfFilters, sizeFilter, sizeDenseUnits, filterOrganization, activation, dropoutProb)
    print("run name = {}".format(run))
    wandb.run.name = run
    Train.runTrain(
        project_name, root, epochs, batch_size, numOfFilters, sizeFilter,
        sizeDenseUnits, batchNormalization, dataAugmentation,
        dropoutProb, activation, filterOrganization, test
    )
    wandb.finish()

if __name__ == '__main__':
    main()
