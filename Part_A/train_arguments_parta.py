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
