import Create_DataLoader
import ModelPreTrained_test
import lightning as pl

'''function to the pre trained model'''
def main():   
    '''
        defining parameteres and loading the dataset
        The values of the parameters are set according to the best model that I had achieved
    ''' 
    batch_size=16
    dataLoader=Create_DataLoader.DatasetLoader_create(root='/kaggle/input/inaturalist-12k/nature_12K_dataset/inaturalist_12K',batch_size=batch_size)
    trainLoader,valLoader,testLoader=dataLoader.data_loaders()
    
    freezed_layers=5
    learning_rate=1e-4
    numOfOutputClasses=10
    epochs=20

    '''creating object of the FineTuningModel class'''
    preTrainedModel=ModelPreTrained_test.FineTuningModel(numOfOutputClasses,freezed_layers,learning_rate)
    
    '''creating trainer object by pytorch lightning'''
    #trainer=pl.Trainer(max_epochs=epochs)
    trainer = pl.Trainer(max_epochs=epochs,strategy="ddp_find_unused_parameters_true",devices=1, enable_progress_bar=True)

    '''exectuing training and validartion step'''
    trainer.fit(preTrainedModel,trainLoader,valLoader)

    '''executing test step and reporting test accuracy'''
    trainer.test(preTrainedModel,testLoader)
    print("Test Accuracy : ",preTrainedModel.test_accuracy/len(testLoader))

if __name__ == '__main__':
    main()