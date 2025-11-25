import torch
from torchvision import transforms
from PIL import Image
from pathlib import Path

from src.Crop_disease_detection.logging import logging
from src.Crop_disease_detection.exceptions import CustomException


class PredictionPipeline:
    def __init__(self, model_path: str, class_names: list):
        self.model_path = model_path
        self.class_names = class_names

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

        self.model = None
        self.load_model()

    def load_model(self):
        try:
            logging.info("Loading trained model for prediction...")
            self.model = torch.load(self.model_path, map_location=self.device)
            self.model.eval()
            logging.info("Model loaded successfully.")

        except Exception as e:
            raise CustomException(e, sys)

    def predict(self, image_path: str):
        try:
            image = Image.open(image_path).convert("RGB")
            image = self.transform(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                outputs = self.model(image)
                _, predicted = outputs.max(1)
                class_index = predicted.item()

            return self.class_names[class_index]

        except Exception as e:
            raise CustomException(e, sys)
