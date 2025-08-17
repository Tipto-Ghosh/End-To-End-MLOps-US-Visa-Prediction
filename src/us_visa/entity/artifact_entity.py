from dataclasses import dataclass


@dataclass
class DataIngestionArtifact:
    train_file_path : str 
    test_file_path : str


@dataclass
class DataValidationArtifact:
    validation_status : bool
    message : str # Data Drift Detected / Not Detected 
    drift_report_file_path : str 


@dataclass
class DataTransformationArtifact:
    transformed_object_file_path : str
    transformed_train_data_file_path : str
    transformed_test_data_file_path : str 


@dataclass
class ClassificationMetricArtifact:
    accuracy_score : float
    f1_score : float
    precision_score : float
    recall_score : float
    

@dataclass
class ModelTrainerArtifact:
    trained_model_file_path : str 
    metric_artifact : ClassificationMetricArtifact
    tuned_model_report_file_path : str