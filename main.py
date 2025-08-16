import sys
from src.us_visa.pipeline.training_pipeline import TrainingPipeline

# Run the training pipeline
training_pipeline_obj = TrainingPipeline()
training_pipeline_obj.run_training_pipeline()