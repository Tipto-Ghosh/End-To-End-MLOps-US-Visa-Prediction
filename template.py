import os
from pathlib import Path
import logging


logging.basicConfig(level = logging.INFO , format = '[%(asctime)s]: %(message)s:')

project_name = "us_visa"

list_of_files = [
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/components/data_ingestion.py",
    f"src/{project_name}/components/data_validation.py",
    f"src/{project_name}/components/data_transformation.py",
    f"src/{project_name}/components/model_trainer.py",
    f"src/{project_name}/components/model_evaluation.py",
    f"src/{project_name}/components/model_pusher.py",
    
    f"src/{project_name}/configuration/__init__.py",
    
    f"src/{project_name}/constants/__init__.py",
    
    f"src/{project_name}/entity/__init__.py",
    f"src/{project_name}/entity/config_entity.py",
    f"src/{project_name}/entity/artifact_entity.py",
    
    f"src/{project_name}/exception/__init__.py",
    f"src/{project_name}/logger/__init__.py",
    
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/utils/main_utils.py",
    

    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/pipeline/training_pipeline.py",
    f"src/{project_name}/pipeline/prediction_pipeline.py",
    
    "config/model.yaml",
    "params.yaml",
    "config/schema.yaml",
    
    "notebooks/EDA.ipynb",
    
    "main.py",
    "app.py",
    
    "requirements.txt",
    "setup.py",
    
    "static/css/style.css",
    "templates/index.html",
    
    "Dockerfile",
    ".dockerignore"
]

# Go to the list items and create's all the folder's and files
for filepath in list_of_files:
    
    filepath = Path(filepath)
    filedir , filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir , exist_ok = True)
        logging.info(f"Creating directory: {filedir} for the file: {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath , "w") as f:
            pass 
        
        logging.info(f"Creating empty file: {filepath}")

    else:
        logging.info(f"{filename} is already exists")