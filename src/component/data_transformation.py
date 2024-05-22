import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object

#why is data class required 
#   By using the @dataclass without using the constructor __init__(), the class (DataTransformationConfig ) accepted the value and assigned to the given variable, so that in this case automatically the 'preprocessor.pkl' file will be created in the 'artifacts' folde
@dataclass
class DataTransformationConfig:
    
    #to create any model and save the model in a file 
    preprocess_obj_file_path=os.path.join('artifact',"preprocessor.pkl")


##this function is to transform the data 
class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
    
    def get_data_transformer_obj(self):
        """_summary_

        this function is for data transformation 
        """
        
        
        try:
            numerical_columns = ["reading_score", "writing_score"]
            categorical_columns= ["gender", "race_ethnicity", "parental_level_of_education", "lunch", "test_preparation_course"]
            
            #pipline that handles scaler 
            numerical_pipline=Pipeline(
                steps=[
                    #how does this handle missing values
                    ("imputer",SimpleImputer(strategy='median')),
                    #scale the vales 
                    ("scaler",StandardScaler())
                ]
            )   
            categorical_pipline=Pipeline(
                steps=[
                    #handle missing values 
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("OneHotEncoder",OneHotEncoder()),
                    #scale the values
                    ("scaler",StandardScaler(with_mean=False))       
                ]         
            )
            logging.info(f"categorical columns: {categorical_columns}"),
            logging.info(f"numerical columns: {numerical_columns}")
            
            
            
            #to combine the numerical and categorical pipline together 
            preprocessor=ColumnTransformer(
                [      #pipline name     #pipeline we specified      #veriables to be used 
                    ("numerical_pipline",numerical_pipline,numerical_columns),
                    ("categorical_pipline",categorical_pipline,categorical_columns)
                ]
                
            )
            return preprocessor
        
        
        except Exception as e:
            raise CustomException(e,sys)
    
    
    
    def initiate_data_transformation(self,train_path,test_path):
          
        try:
            
            #read the file path
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            logging.info("read train and test complete")
            
            logging.info("obtaining preprocessing object")
            
            #initialise the transform function
            preprocessing_obj=self.get_data_transformer_obj()
            
            target_column_name="math_score"
            numerical_columns = ["writing_score", "reading_score"]
            
            #get your target
            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]
            
            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]
            
            logging.info(f"Applying preprocessing object on training dataframe and testing dataframe.")
            
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)
            #convert to an array
            #what does this do ???
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            
            train_arr = np.asarray(train_arr)
            test_arr = np.asarray(test_arr)
             
            logging.info(f"Saved preprocessing object.")
            #saving the pkl file (function is in uti)
            save_object(
                file_path=self.data_transformation_config.preprocess_obj_file_path,
                obj=preprocessing_obj
            )
            
            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocess_obj_file_path
            )
        except Exception as e:
            raise CustomException(e,sys)
        
        
            
        