import argparse

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
