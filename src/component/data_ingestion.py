import os 
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.component.data_transformation import DataTransformation
from src.component.data_transformation import DataTransformationConfig

from src.component.model_trainer import ModeltrainerConfig
from src.component.model_trainer import ModelTrainer
import numpy as np


@dataclass
class DataingestionConfig:
    train_data_path: str=os.path.join('artifact',"train.csv")
    test_data_path: str=os.path.join('artifact',"test.csv")
    raw_data_path: str=os.path.join('artifact',"data.csv")
    
class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataingestionConfig()
        
    def initiate_data_ingestion(self):
        logging.info("enter the data ingestion component")
        
        try:
            #read data from source 
            df=pd.read_csv('notebook/data/stud.csv')
            logging.info("read the dataset as dataframe")
            
            #make a directory to store the data (artifact folder)
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            
            #convert the raw data into csv file 
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            
            
            logging.info("train test split initialted ")
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)
            
            
            # saving the train and test inside artifact folder 
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            
            logging.info("ingestion of the data is completed")
            
            return(
                #basicall returning the test data path and train data path 
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
                
            )
        except Exception as e:
            raise CustomException(e,sys)
        


if __name__=="__main__":
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()
    
    data_transformation=DataTransformation()
    
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)
    
    modeltrainer=ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_array=train_arr,test_array=test_arr))
    
    
    