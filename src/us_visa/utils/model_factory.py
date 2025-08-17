import os
import sys 
import yaml
from importlib import import_module
from typing import Any, Dict, Tuple, List

import numpy as np 
import pandas as pd 

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from dataclasses import dataclass

from src.us_visa.logger import logging
from src.us_visa.exception import UsVisaException
from src.us_visa.utils.main_utils import read_yaml_file, save_yaml_file, load_object

@dataclass
class BestModelDetail:
    """
    Dataclass to store information about the best model.
    
    Attributes:
        best_model (Any): Trained model object (fitted with best parameters)
        best_score (float): Test accuracy score of the best model
        best_params (Dict): Best hyperparameters found during tuning
        model_name (str): Name of the model
        module_name (str): Module path of the model class
    """
    best_model : Any
    best_score : float
    best_params : Dict
    model_name : str
    module_name : str
    
    
class ModelFactory:
    """
    ModelFactory class is responsible for:
        1. Reading models and hyperparameters from a YAML configuration file.
        2. Initializing model objects dynamically.
        3. Performing hyperparameter tuning (GridSearchCV / RandomizedSearchCV).
        4. Evaluating models on training and test datasets.
        5. Saving all model performance and best parameters to a YAML report.
        6. Returning the best model object based on highest test accuracy.
    """
    
    def __init__(self , model_config_path: str , tuned_model_report_path: str) -> None:
        """
        Initialize ModelFactory instance.
        
        Args:
            model_config_path (str): Path to model YAML configuration file.
            tuned_model_report_path (str): Path to save all tuned models report.
        """
        
        try:
            self.model_config_path = model_config_path
            self.tuned_model_report_file_path = tuned_model_report_path
            self.model_config = None # content will be the content of model.yaml
            self.tuned_model_report = {} # store the result of all tuned model after gridsearch cv
            
        except Exception as e:
            raise UsVisaException(e , sys) from e
    
    def read_model_config(self) -> Dict:
        """
        Reads the model YAML configuration file and returns a dictionary.
        
        Returns:
            Dict: Dictionary containing all models, their modules, default parameters, 
                  and hyperparameter search grids. Also save the model_config
        """
        try:
            if self.model_config is None:
                self.model_config = read_yaml_file(self.model_config_path)
            return self.model_config
        
        except Exception as e:
            raise UsVisaException(e , sys) from e 
    
    def initialize_model(self, model_info: Dict) -> Tuple[str, object, Dict]:
        """
        Dynamically imports the module and class, then instantiates the model object with default parameters.

        Args:
            model_info (Dict): Dictionary containing:
                - "module": module path (e.g., "sklearn.ensemble")
                - "class": class name (e.g., "RandomForestClassifier")
                - "params": default parameters for initialization
                - "search_param_grid": hyperparameter grid for tuning

        Returns:
            Tuple[str, object, Dict]: 
                - model_name (str)
                - instantiated model object (object)
                - hyperparameter grid (Dict)
        """
        
        
        try:
            module_name = model_info['module']
            class_name = model_info['class']
            model_name = class_name  
            params = model_info.get('params', {})
            param_grid = model_info.get('search_param_grid', {})

            # Dynamically import module and get class
            module = import_module(module_name)
            model_class = getattr(module, class_name)

            # Instantiate model with default parameters
            model_obj = model_class(**params)

            return model_name, model_obj, param_grid
        except Exception as e:
            raise UsVisaException(e , sys) from e 
    
    def tune_model(self , X_train: np.ndarray , y_train: np.ndarray , model_name: str , model_obj: object , param_grid: Dict) -> Dict:
        """
        Performs GridSearchCV / RandomizedSearchCV to find the best hyperparameters on training data.
        
        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training target
            model_name (str): Name of the model
            model_obj (object): Instantiated model object
            param_grid (Dict): Hyperparameter search space
        
        Returns:
            Dict: Dictionary containing 'best_params', 'train_score', and fitted 'model_obj'
        """
        
        #Task: Take a model object(sklearn object) and model_obj params , and train on data
        
        logging.info(f"Starting hyperparameter tuning for model: {model_name}")
        
        # read the search strategy from the yaml content
        search_config = self.model_config.get("grid_search" , {})
        
        # read the search method(GridSearchCV / RandomizedSearchCV) if not given then defeault is GridSearchCV
        search_class_name = search_config.get("class", "GridSearchCV")
        
        # read class sklearn import module part
        search_module_name = search_config.get("module", "sklearn.model_selection")
        
        # read the search params dict
        search_params = search_config.get("params", {"cv": 3, "verbose": 2, "n_jobs": -1}) # if not set then use defult
        
        # import search class from sklearn.model_selection (GridSearchCV / RandomizedSearchCV)
        SearchClass = getattr(__import__(search_module_name, fromlist = [search_class_name]), search_class_name)
        
        # make the object of search class(object of GridSearchCV / RandomizedSearchCV)
        # check which tuning class needs to use
        if search_class_name == "RandomizedSearchCV":
            search_obj = SearchClass(
                estimator = model_obj,
                param_distributions = param_grid, 
                **search_params
            )
        else:  
            search_obj = SearchClass(
                estimator = model_obj,
                param_grid = param_grid,
                **search_params
            )
        
        logging.info(f"Starting tuning [{model_name}] Using search strategy: {search_class_name}")
        
        
        # fit the (GridSearchCV / RandomizedSearchCV) on train data
        search_obj.fit(X_train , y_train)
        
        # find the best estimator , params and train_score
        best_model = search_obj.best_estimator_
        best_params = search_obj.best_params_
        train_score = accuracy_score(y_train, best_model.predict(X_train))
        
        logging.info(
          f"[{model_name}] => Completed tuning | Best Params: {best_params}, Train Accuracy: {train_score:.4f}"
        )
        
        return {
            "model_name": model_name,
            "best_model": best_model,
            "best_params": best_params,
            "train_score": train_score
        }
        
    def evaluate_model(self, model_obj: object, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Evaluates a trained model on a dataset and returns classification metrics.
        
        Args:
            model_obj (object): Trained model object
            X (np.ndarray): Features
            y (np.ndarray): Target
        
        Returns:
            Dict: Dictionary containing 'accuracy', 'f1', 'precision', 'recall'
        """
        
        try:
           logging.info(f"started evaluating from evaluate_model method of ModeFactory [{model_obj.__class__.__name__}]")
           
           # predict on given X data
           y_pred = model_obj.predict(X)
           
           # Calculate metrics
           accuracy = accuracy_score(y , y_pred)
           f1 = f1_score(y , y_pred , average = "weighted")  
           precision = precision_score(y , y_pred , average = "weighted")
           recall = recall_score(y , y_pred , average = "weighted")
           metrics = {
                "accuracy": accuracy,
                "f1": f1,
                "precision": precision,
                "recall": recall
           }
           
           logging.info(f"Evaluation metrics for model [{model_obj.__class__.__name__}]: {metrics}")
           return metrics
       
        except Exception as e:
           raise UsVisaException(e , sys) from e  
    
    def run_model_factory(self , X_train: np.ndarray , y_train: np.ndarray , X_test: np.ndarray , y_test: np.ndarray) -> Dict[str, Dict]:
        """
        Runs hyperparameter tuning for all models from YAML and evaluates train & test performance.
        Saves results to tuned_model_report_path.
        
        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training target
            X_test (np.ndarray): Test features
            y_test (np.ndarray): Test target
        
        Returns:
            Dict[str, Dict]: Dictionary containing all models' train/test metrics and best parameters
        """
        try:
            logging.info("Entered into run_model_factory method of class ModelFactory")
            
            # 1. read the model.yaml file
            self.read_model_config()
            logging.info(f"after reading the yaml file len of self.model_config: [{len(self.model_config)}]")
            
            # get the model info part from model config
            models_info = self.model_config.get("model_selection" , {})
            logging.info(f"Len of models_info: [{len(models_info)}]")
            
            # 2. clear tuned model report if any
            self.tuned_model_report = {}
            logging.info("clear tuned model report")
            
            # 3. read the yaml content and train each model
            for model_key , model_info in models_info.items():
                # initialize the model
                model_name, model_obj, param_grid = self.initialize_model(model_info)
                logging.info(f"initialized model [name = {model_name}]")
                
                # tune the model
                tuned_result = self.tune_model(
                    X_train = X_train, 
                    y_train = y_train,
                    model_name = model_name,
                    model_obj = model_obj,
                    param_grid = param_grid
                )
                
                # evaluate the model
                train_metrics = self.evaluate_model(tuned_result["best_model"], X_train, y_train)
                test_metrics = self.evaluate_model(tuned_result["best_model"], X_test, y_test)
                
                # Store results in report
                self.tuned_model_report[model_name] = {
                    "best_params": tuned_result["best_params"],
                    "train_score": tuned_result["train_score"],
                    "train_metrics": train_metrics,
                    "test_metrics": test_metrics,
                    "module_name": model_info["module"]
                }
            
            # save the report
            save_yaml_file(
                file_path = self.tuned_model_report_file_path,
                data = self.tuned_model_report 
            )
            logging.info(f"Tuned model report saved at {self.tuned_model_report_file_path}")
            
            return self.tuned_model_report
        
        except Exception as e:
            raise UsVisaException(e , sys) from e 
    
    def get_best_model(self ) -> BestModelDetail:
        """
        Returns the best model object based on highest test accuracy.If tuned report exists, loads it; otherwise, runs model factory.
        
        Returns:
            BestModelDetail: Dataclass object containing best_model_object, best_params, best_score, model_name
        """
        try:
            if not self.tuned_model_report:
                raise UsVisaException("Tuned model report is empty. Please run run_model_factory first.", sys)
            
            # find the best model based on highest test accuracy
            best_score = -1
            best_model_detail = None 
            
            for model_name , result in self.tuned_model_report.items():
                
                test_accuracy = result["test_metrics"]["accuracy"]
                if test_accuracy > best_score:
                    best_score = test_accuracy
                    
                    best_model_detail = BestModelDetail(
                        best_model = result["best_model"],
                        best_score = test_accuracy,
                        best_params = result["best_params"],
                        model_name = result.get("model_name" , model_name),
                        module_name = result.get("module_name", "")
                    )
            
            if best_model_detail is None:
                raise UsVisaException("No model found in tuned model report.", sys)
            
            logging.info(f"Best model: {best_model_detail.model_name} | Test Accuracy: {best_score:.4f}")
            return best_model_detail
        
        
        except Exception as e:
            raise UsVisaException(e , sys) from e 