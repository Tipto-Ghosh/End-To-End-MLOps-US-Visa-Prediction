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