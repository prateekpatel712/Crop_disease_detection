import os
import sys
from pathlib import Path
from src.Crop_disease_detection.logging import logging
from src.Crop_disease_detection.exceptions import CustomException 

class DataIngestionConfig:
    def __init__(self):
        self.data_ingestion_url = "URL will be replaced in colab"
        self.raw_data_dir = Path("data/raw") 
        self.processed_data_dir = Path("data/processed")

    def download_dataset(self):
        try:
            logging.info("Starting dataset download...")

            self.raw_data_dir.mkdir(parents=True, exist_ok=True)
            logging.info(f"Raw data directory created at: {self.raw_data_dir}")

            self.zip_file_path = self.raw_data_dir / "dataset.zip"
            logging.info(f"Zip file will be saved at: {self.zip_file_path}")

            logging.info("Dataset URL is set. Actual download will happen in Colab.")
            # Example download code (to be added in Colab)
            # response = requests.get(self.data_ingestion_url)
            # with open(self.zip_file_path, "wb") as f:
            #     f.write(response.content)

        except Exception as e:
            logging.error("Error occurred during dataset download.")
            raise CustomException(e, sys)

    def extract_dataset(self):
        try:
            logging.info("Preparing to extract dataset...")

            self.processed_data_dir.mkdir(parents=True, exist_ok=True)
            logging.info(f"Processed data directory created at: {self.processed_data_dir}")

            logging.info("Extraction logic will run in Colab when the ZIP file is downloaded.")
            # Example extraction code (to be added in Colab)
            # with zipfile.ZipFile(self.zip_file_path, 'r') as zip_ref:
            #     zip_ref.extractall(self.processed_data_dir)

        except Exception as e:
            logging.error("Error occurred during dataset extraction.")
            raise CustomException(e, sys)

    def split_dataset(self):
        try:
            logging.info("Preparing to split dataset into train and test sets...")

            self.extracted_base_path = self.processed_data_dir / "PlantVillage"
            logging.info(f"Expected extracted dataset folder: {self.extracted_base_path}")

            if not self.extracted_base_path.exists():
                logging.error("Extracted dataset folder not found.")
                raise CustomException("Extracted dataset not found. Extraction must run first.", sys)

            class_folders = [folder for folder in self.extracted_base_path.iterdir() if folder.is_dir()]
            logging.info(f"Found {len(class_folders)} class folders.")

            train_dir = self.processed_data_dir / "train"
            test_dir = self.processed_data_dir / "test"

            train_dir.mkdir(exist_ok=True)
            test_dir.mkdir(exist_ok=True)

            logging.info(f"Train directory: {train_dir}")
            logging.info(f"Test directory: {test_dir}")

            # Example split loop (to be added in Colab)
            # for class_folder in class_folders:
            #     images = list(class_folder.glob("*"))
            #     train_imgs, test_imgs = train_test_split(images, test_size=0.2, random_state=42)
            #
            #     class_train_dir = train_dir / class_folder.name
            #     class_test_dir = test_dir / class_folder.name
            #
            #     class_train_dir.mkdir(exist_ok=True)
            #     class_test_dir.mkdir(exist_ok=True)
            #
            #     for img in train_imgs:
            #         img.rename(class_train_dir / img.name)
            #
            #     for img in test_imgs:
            #         img.rename(class_test_dir / img.name)

        except Exception as e:
            logging.error("Error occurred during train/test split.")
            raise CustomException(e, sys)

    def initiate_data_ingestion(self):
        try:
            self.download_dataset()
            self.extract_dataset()
            self.split_dataset()
            logging.info("Data ingestion pipeline completed successfully.")
        except Exception as e:
            raise CustomException(e, sys)
