from sklearn.ensemble import RandomForestClassifier
from src.us_visa.utils.main_utils import read_yaml_file
from src.us_visa.constants import SCHEMA_FILE_PATH
from typing import Dict

# print(SCHEMA_FILE_PATH)

# print(read_yaml_file(SCHEMA_FILE_PATH)['transform_columns'])

# import pandas as pd 

# df = pd.DataFrame({"id" : [1,2,3] , "name" : ["a" , 'b' , 'c']})

# print(df.columns.to_list())

# import numpy as np 
# print(np.__version__)

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from src.us_visa.utils.model_factory import ModelFactory


X, y = make_classification(
    n_samples = 200,    
    n_features = 5,
    n_informative = 3,
    n_redundant = 0,
    n_classes = 2,
    random_state = 42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.2, random_state = 42
)


sample_model_yaml_path = "sample_model.yaml"
test_report_file_path = "test.yml"

model_fac = ModelFactory(
    model_config_path = sample_model_yaml_path,
    tuned_model_report_path = test_report_file_path
)

tuned_report = model_fac.run_model_factory(X_train = X_train , y_train = y_train , X_test = X_test , y_test = y_test)

# print("=" * 30)
# print(tuned_report)
# print("=" * 30)

best_model_detail = model_fac.get_best_model()

module_name = best_model_detail.module_name
class_name = best_model_detail.model_name
best_params = best_model_detail.best_params

print("Best Model:", best_model_detail.model_name)
print("Best Parameters:", best_model_detail.best_params)
print("Test Accuracy:", best_model_detail.best_score)

# print(model_fac.tuned_model_report)

print("=" * 40)

print(type(best_model_detail.best_model))

print("=" * 40)

model_obj = best_model_detail.best_model

print("Best Model:", best_model_detail.model_name)
print("Best Parameters:", best_model_detail.best_params)
print("Test Accuracy:", best_model_detail.best_score)
print("=" * 40)

model_obj.fit(X_train , y_train)

y_pred = model_obj.predict(X_test)

from sklearn.metrics import accuracy_score, f1_score

print("=" * 40)
print("=" * 40)

print("Test Accuracy:", accuracy_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred, average="weighted"))
print("=" * 40)
print("=" * 40)
