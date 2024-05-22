import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models


@dataclass
class ModeltrainerConfig:
    trained_model_file_path=os.path.join("artifact","model.plk")
    
class ModelTrainer:
    def __init__(self):
        #by having this variable we will be able to get the path name 
        self.model_trainer_config=ModeltrainerConfig()
        
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("spliting training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            logging.info("did it get here")
            #creating a dictionary of models
            models = {
                "Random Forest" : RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boost": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "k-Neighbor Classifier": KNeighborsRegressor(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Classifier": CatBoostRegressor(verbose=False),
                "AdaBoost Classifier": AdaBoostRegressor(),
            }
            models_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models)
            
            ## to get best model score in a list
            best_model_score = max(sorted(models_report.values()))


            ## To get best model name from dict
            best_model_name = list(models_report.keys())[
                list(models_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]
            
            if best_model_score<0.6:
                raise CustomException("no best model found")
            
            logging.info(f"best found model in both training and testing datset")
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            
            predicted=best_model.predict(X_test)
            
            r2_square = r2_score(y_test,predicted)
            return r2_square
        
            
            
        except Exception as e:
            raise CustomException(e,sys)
        
        