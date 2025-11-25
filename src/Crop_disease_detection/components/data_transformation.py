import os
import sys
from pathlib import Path
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from src.Crop_disease_detection.logging import logging
from src.Crop_disease_detection.exceptions import CustomException

class DataTransformationConfig:
    def __init__(self):
        self.batch_size = 32
        self.image_size = (224, 224)

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

        self.train_transform = transforms.Compose([
            transforms.Resize(self.config.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

        self.test_transform = transforms.Compose([
            transforms.Resize(self.config.image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

    def initialize_dataloaders(self, train_dir: Path, test_dir: Path):
        try:
            logging.info("Initializing datasets with transforms...")

            train_dataset = ImageFolder(root=train_dir, transform=self.train_transform)
            test_dataset = ImageFolder(root=test_dir, transform=self.test_transform)

            logging.info(f"Train dataset size: {len(train_dataset)}")
            logging.info(f"Test dataset size: {len(test_dataset)}")

            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=2
            )

            test_loader = DataLoader(
                test_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=2
            )

            return train_loader, test_loader, train_dataset.classes

        except Exception as e:
            logging.error("Error during data transformation.")
            raise CustomException(e, sys)
