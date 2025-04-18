import wandb
import warnings
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
from train_arguments_parta import arguments

warnings.filterwarnings("ignore")

''' attach the forward method and helper functions to the CNN class'''
Q1_CNNmodel.CNN.forward = Q1_ForwardPass.forward
Q1_CNNmodel.CNN.activationFunction = Q1_HelperFunctions.activationFunction
Q1_CNNmodel.CNN.filterSizeCalculator = Q1_HelperFunctions.filterSizeCalculator

'''login to wandb to generate plot'''
wandb.login()

'''setting the device to gpu if it is available'''
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

'''class to support the train_parta.py execution'''
class Train:
    def runTrain(project_name, root, epochs, batch_size, numOfFilters, sizeFilter, sizeDenseUnits, batchNormalization, dataAugmentation, dropoutProb, activation, filterOrganization, test):
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

'''main driver function'''
def main():
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

    '''call to argument function imported from train_arguments_parta.py file to get the arguments'''
    args = arguments()

    '''checking if a particular argument is passed through command line or not and updating the values accordingly'''
    if args.wandb_project is not None: project_name = args.wandb_project
    if args.wandb_entity is not None: entity_name = args.wandb_entity
    if args.root is not None: root = args.root
    if args.epochs is not None: epochs = args.epochs
    if args.batch is not None: batch_size = args.batch
    if args.filter is not None: numOfFilters = args.filter
    if args.filter_size is not None: sizeFilter = args.filter_size
    if args.neurons is not None: sizeDenseUnits = args.neurons
    if args.batch_normal is not None: batchNormalization = args.batch_normal
    if args.data_augment is not None: dataAugmentation = args.data_augment
    if args.dropout is not None: dropoutProb = args.dropout
    if args.activation is not None: activation = args.activation
    if args.filter_org is not None: filterOrganization = args.filter_org
    if args.test is not None: test = args.test

    '''initializing to the project'''
    wandb.init(project=project_name, entity=entity_name)

    '''calling the functions with the parameters'''
    run = "EP_{}_FIL_{}_FILSIZE_{}_FCSIZE_{}_FILORG_{}_AC_{}_DRP_{}".format(epochs, numOfFilters, sizeFilter, sizeDenseUnits, filterOrganization, activation, dropoutProb)
    print("run name = {}".format(run))
    wandb.run.name = run

    Train.runTrain(project_name, root, epochs, batch_size, numOfFilters, sizeFilter,
                   sizeDenseUnits, batchNormalization, dataAugmentation,
                   dropoutProb, activation, filterOrganization, test)

    wandb.finish()

if __name__ == '__main__':
    main()
