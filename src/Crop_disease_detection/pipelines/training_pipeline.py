import sys
from src.Crop_disease_detection.logging import logging
from src.Crop_disease_detection.exceptions import CustomException

from src.Crop_disease_detection.components.data_ingestion import DataIngestionConfig
from src.Crop_disease_detection.components.data_transformation import (
    DataTransformation, DataTransformationConfig
)
from src.Crop_disease_detection.components.model_trainer import (
    ModelTrainer, ModelTrainerConfig
)

class TrainingPipeline:
    def __init__(self):
        self.ingestion = DataIngestionConfig()
        self.transform_config = DataTransformationConfig()
        self.model_config = ModelTrainerConfig()

    def run(self):
        try:
            logging.info("=== Starting Training Pipeline ===")

            # Step 1: Data Ingestion
            self.ingestion.download_dataset()
            self.ingestion.extract_dataset()
            self.ingestion.split_dataset()

            # Extract train/test folders
            train_dir = self.ingestion.processed_data_dir / "train"
            test_dir = self.ingestion.processed_data_dir / "test"

            # Step 2: Data Transformation (Get Dataloaders)
            transformer = DataTransformation(self.transform_config)
            train_loader, test_loader, class_names = transformer.initialize_dataloaders(
                train_dir, test_dir
            )

            # Step 3: Model Training
            trainer = ModelTrainer(self.model_config, num_classes=len(class_names))
            model = trainer.train(train_loader, test_loader)

            logging.info("=== Training Pipeline Completed Successfully ===")

        except Exception as e:
            raise CustomException(e, sys)
