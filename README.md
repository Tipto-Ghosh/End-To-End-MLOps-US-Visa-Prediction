## End-To-End-MLOps-US-Visa-Prediction


- Dataset Link: https://www.kaggle.com/datasets/moro23/easyvisa-dataset


## Workflow
---
### Data Ingestion workflow
1. constants
2. config_entity
3. artifact_entity
4. data access(connection with the database)
5. update the data_ingestion component
6. add data_ingestion to the training pipeline

---
### Data Validation workflow
1. constants
2. config_entity
3. artifact_entity
4. update the data_validation component
5. add data_validation to the training pipeline

---
### Data Transformation workflow
1. update constants
2. update config_entity
3. update artifact_entity
4. update the data_transformation component
5. add data_transformation to the training pipeline