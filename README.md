# DA6401 Assignment 2

## Fetching all the code files
You need to first clone the repository in the Github containing all the files.
```
git clone https://github.com/DEBASMITA2702/DL_Assignment2.git
```
Then change into the code directory.
```
cd DL_Assignment2
```
Ensure you are in the correct directory before proceeding further.


## Setting up the platform and environment
- ### Local machine
  If you are running the code on a local machine, then you need to have python installed in the machine and pip command added in the environemnt variables.
  You can execute the following command to setup the environment and install all the required packages
  ```
  pip install -r requirements.txt
  ```
- ### Google colab/Kaggle
  If you are using google colab platform or kaggle, then you just need to run the following code
  ```
  pip install wandb torch lightning pytorch_lightning matplotlib torchvision torchmetrics
  ```
This step will setup the environment required before proceeding.


## Project Parts
The project deals with working with Convolutional Neural Networks(CNN). It is divided into the two following parts:
- ### Part A
  This part has a CNN model trained from scratch and corresponding train and test files implemented. The codes to run this part are present in the Part_A directory.
- ### Part B
  This part has a pre trained model called GoogLeNEt, which is fine tuned to work for the given dataset. The codes to run this part are present in the Part_B directory.


## Loading the dataset
The dataset needs to be placed in the home directory, thst is, in the 'DL_Assignment2' directory.
If the directory is placed somewhere else, then to execute the files related to this project, you need to specify the absolute path of the root of the dataset.
For example, if you want to run train_parta.py with the dataset located somewhere else other than the home directory then you need to run the following: 
```
python train_parta.py --root <absoulte_path_of_dataset>
```
The same needs to be executed for train_partb.py

#### Note
Do not give the paths for the train and val folders seperately. Just pass the absolute path of the root directory of the dataset, i.e. the directory ```inaturalist_12K``` as the argument. The train and val folders will be seperately handled inside the code itself.


## Part A
Make sure to change the directory by the command
```
cd Part_A
```

To train the model, you need to compile and execute the [train.py](https://github.com/DEBASMITA2702/DL_Assignment2/blob/main/Part_A/train_parta.py) file, and pass additional arguments if and when necessary.\
It can be done by using the command:
```
python train_parta.py
```
By the above command, the model will run with the default configuration set inside the code.\
To customize the run, you need to specify the parameters like ```python train_parta.py <*args>```\
For example,
```
python train_parta.py -e 20 -b 64 --filter 256 
```

The arguments supported are :
|           Name           | Default Value | Description                                                               |
| :----------------------: | :-----------: | :------------------------------------------------------------------------ |
| `-wp`, `--wandb_project` | Debasmita-DA6401-Assignment-2 | Project name used to track experiments in Weights & Biases dashboard      |
|  `-we`, `--wandb_entity` |     cs24m015-indian-institute-of-technology-madras    | Wandb Entity used to track experiments in the Weights & Biases dashboard. |
|     `-r`,`--root`        |../inaturalist_12K |Absolute path of the dataset                                         |
|     `-e`, `--epochs`     |       5      | Number of epochs to train neural network.                                 |
|   `-b`, `--batch`        |       32       | Batch size used to train neural network.                                  |
|   `-f`,`--filter`        |       32      | Number of filters in the first convolution layer                          |
|   `-fs`,`--filter_size`  |        7      | Dimension of the filters                                                  |
|    `-n`,`--neurons`      |     64      | Number of neurons in the fully connected layer                            |
|    `-bn`,`--batch_normal`|     Yes       | choices: ['Yes','No']                                                     |
|   `-da`,`--data_augment` |      No       | choices: ['Yes','No']                                                     |
|   `-d`,`--dropout`       |       0.4       | Percentage of dropout in the network                                      |
|   `-a`,`--activation`    |     Mish      | Activation function in the activation layers                              |
|   `-fo`,`--filter_org`   |      same     | Organization of the filters across the layers                             |
|   `-t`,`--test`          |       0       | choices: [0,1]                                                            |

The arguments can be changed as per requirement through the command line.
  - If prompted to enter the wandb login key, enter the API key in the interactive command prompt.

## Testing the model
To test the model, you need to set the test argument as '1'. For example
```
python train_parta.py -t 1
```
This will run the model with default parameters and print the test accuracy.

## Code Organization for Part A
For Question 1, we have 3 files.
- Q1_CNNmodel file : I have defined a CNN class inside this file, in which, the init constructor method initializes a the object with different parameters like size of filters, number of filters, filter organization, neurons in the fully connected layer, etc. Also the 5 convolution-activation-maxpool blocks are defined here, then the fully connected layer and finally the output layer.
- Q1_HelperFunctions file : This file has two methods.
  - filterSizeCalculator : This function calculates and returns the filter size after apply a certain filter organization scheme.
  - activationFunction : This function returns an activation function object based on the parameter passed.
- Q1_ForwardPass file  : Here, we have a method 'forward' which implements forward propagation on the model defined by the CNN class, and it returns the same object after applying the forward propagation.

For Question 2, we have 5 files.
- Q2_LoadDataset file : This file contains the core functionality (initialization, image transformations, and the method to load and split the dataset into training and validation sets).
- Q2_CreateDataLoader file : This file imports the core functionality from LoadDataset.py and adds the data_loaders method to create and return DataLoader objects.
- Q2_LightningModel_Base file : It contains the functionalities like initialization, optimizer configuration, training and validation steps.
- Q2_LightningModel_EpochLogs file : This file extends the core class by adding the epoch-end logging methods.
- Q2_main file : It is the driver code that initializes and runs sweep for CNN hyperparameter optimization using PyTorch Lightning. It integrates data loading, augmentation, early stopping, and model training for the inaturalist_12K dataset.

For Question 4, we have 2 files.
- Q4_TestModel file : Contains all the original imports and the 'TestBestModel' class with its testAccuracy() method, that calculates the test accuracy.
- Q4_RunTestModel file : It is like a driver code that simply imports the TestBestModel from Q4_TestModel.py and then calls the testAccuracy() function to run the complete test pipeline.

## Part B
Make sure to change the directory by the command
```
cd Part_B
```

To train the model, you need to compile and execute the [train.py](https://github.com/DEBASMITA2702/DL_Assignment2/blob/main/Part_B/train_partb.py) file, and pass additional arguments if and when necessary.\
It can be done by using the command:
```
python train_partb.py
```
By the above command, the model will run with the default configuration as specified inside my code. Since there was a positive corelation in Part A co-relation summary table, I tried running with higher number of epochs in Part B.\
To customize the run, you need to specify the parameters like ```python train_partb.py <*args>```\
For example,
```
python train_partb.py -e 20 -b 64 --freezed 10 
```

The arguments supported are :
|           Name           | Default Value | Description                                                               |
| :----------------------: | :-----------: | :------------------------------------------------------------------------ |
|     `-r`,`--root`        |../inaturalist_12K |Absolute path of the dataset                                         |
|     `-e`, `--epochs`     |       20      | Number of epochs to train neural network.                                 |
|   `-b`, `--batch`        |       32       | Batch size used to train neural network.                                  |
|   `-lr`,`--learning`     |    1e-4       | Learning rate to train the model                                          |
|   `-fr`,`--freezed`      |      5        | Number of layers freezed from the beginning                               |
|   `-t`,`--test`          |       0       | choices: [0,1]                                                            |

The arguments can be changed as per requirement through the command line.
  - If prompted to enter the wandb login key, enter the API key in the interactive command prompt.

## Testing the model
To test the model, you need to specify the test argument as 1. For example
```
python train_partb.py -t 1
```
This will run the model with default parameters and print the test accuracy.

## Code Organization for Part A
For Question 3, we have the 6 following files.
- LoadDataset file : It contains the DatasetLoader class (with the data splitting logic).
- Create_DataLoader file : It defines a new class 'DatasetLoader_create' with method data_loaders' that creates, returns, and encapsulates the data loaders. It imports 'DatasetLoader' from LoadDataset.py file.
- ModelPreTrained_Base file : This file contains the base class that sets up the pre-trained model, its forward pass, and the optimizer configuration.
- ModelPreTrained_train_val file : It defines a new class 'FineTuningModelTrainVal' which inherits from the base class and adds the training and validation step methods along with the related epoch-end methods.
- ModelPreTrained_test file : It defines the final class 'FineTuningModel' that inherits from the'FineTuningModelTrainVal' class of ModelPreTrained_train_val.py and implements the test_step method.
- Question3 file : It is the driver code for Part B, that fine-tunes the pretrained ImageNet model using PyTorch Lightning for multi-class classification on the inaturalist_12K dataset. It handles data loading, model training, validation, and testing with predefined best hyperparameters.

## Additional features
The following features are also supported
  - If you need some clarification on the arguments to be passed, then you can do
    ```
    python train_parta.py --help
    ```

## Links
[Wandb Report](https://wandb.ai/cs24m015-indian-institute-of-technology-madras/Debasmita-DA6401-Assignment-2/reports/DA6401-Assignment-2-Debasmita-Mandal-CS24M015--VmlldzoxMjI2MTg4NA?accessToken=j1x7849boqgz0si28cciak6of1sh0xbxeehc18143x16y4mi58sqv0415xi311x6)
