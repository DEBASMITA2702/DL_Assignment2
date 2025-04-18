import torch
import Create_DataLoader
import ModelPreTrained_test
import lightning as pl
import wandb
import warnings
import argparse

warnings.filterwarnings("ignore")

'''login to wandb to generate plot'''
wandb.login()

'''setting the device to gpu if avaiable'''
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

'''helper class to execute fine-tuning and optional testing'''
class PreTrained:
    def run(root, epochs, batch_size, learning_rate, freezed, test):
        '''
        Parameters:
            root : absolute path of the dataset
            epochs : number of epochs to run
            batch_size : batch size to split the dataset
            learning_rate : learning rate used to train the model
            freezed : number of layers freezed starting from the input layer
            test : boolean variable denoting whether or not to test the model 
        Returns :
            None
        Function:
            Executes the Fine Tuning on the model
        '''

        '''loads dataset'''
        dataLoader = Create_DataLoader.DatasetLoader_create(root=root, batch_size=batch_size)
        trainLoader, valLoader, testLoader = dataLoader.data_loaders()
        
        '''setting number of output classes to 10'''
        numOfOutputClasses = 10

        '''creating the object of the class and a trainer on it'''
        preTrainedModel = ModelPreTrained_test.FineTuningModel(numOfOutputClasses, freezed, learning_rate)

        trainer = pl.Trainer(max_epochs=epochs)

        '''executing train and validation steps'''
        trainer.fit(preTrainedModel, trainLoader, valLoader)

        '''if prompted then executing test step'''
        if test == 1:
            trainer.test(preTrainedModel, testLoader)
            print("Test Accuracy : ", preTrainedModel.test_accuracy / len(testLoader))


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
    commandLineArgument.add_argument('-r','--root', help="Absolute path of the dataset")
    commandLineArgument.add_argument('-e','--epochs', type=int, help="Number of epochs to train neural network")
    commandLineArgument.add_argument('-b','--batch', type=int, help="Batch size to divide the dataset")
    commandLineArgument.add_argument('-lr','--learning', type=float, help="Learning rate to train the model")
    commandLineArgument.add_argument('-fr','--freezed', type=int, help="Number of layers freezed from the beginning")
    commandLineArgument.add_argument('-t','--test', type=int, choices=[0,1], help="choices: [0,1]")
    
    return commandLineArgument.parse_args()

'''main driver function'''
def main():
    '''default values of each of the hyperparameter. it is set according to the config of my best model'''
    epochs = 20
    batch_size = 32
    learning_rate = 1e-4
    freezed = 5
    test = 0
    root = '/kaggle/input/inaturalist-12k/nature_12K_dataset/inaturalist_12K'

    '''call to argument function to get the arguments'''
    args = arguments()

    '''checking if a particular argument is passed through command line or not and updating the values accordingly'''
    if args.epochs is not None:
        epochs = args.epochs
    if args.batch is not None:
        batch_size = args.batch
    if args.learning is not None:
        learning_rate = args.learning
    if args.freezed is not None:
        freezed = args.freezed
    if args.test is not None:
        test = args.test
    if args.root is not None:
        root = args.root

    '''calling the run method with the parameters'''
    PreTrained.run(root, epochs, batch_size, learning_rate, freezed, test)
    wandb.finish()

if __name__ == '__main__':
    main()
