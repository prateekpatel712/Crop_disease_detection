import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from torchvision import models
from src.Crop_disease_detection.logging import logging
from src.Crop_disease_detection.exceptions import CustomException

class ModelTrainerConfig:
    def __init__(self):
        self.num_epochs = 10
        self.learning_rate = 1e-4
        self.model_save_path = Path("artifacts/model.pth")

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig, num_classes: int):
        self.config = config
        self.num_classes = num_classes

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {self.device}")

        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, self.num_classes)
        self.model = self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)

    def train_one_epoch(self, train_loader):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_loss = running_loss / total
        epoch_accuracy = correct / total

        return epoch_loss, epoch_accuracy

    def validate(self, test_loader):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        val_loss = running_loss / total
        val_accuracy = correct / total

        return val_loss, val_accuracy

    def train(self, train_loader, test_loader):
        logging.info("Starting model training...")

        best_accuracy = 0.0

        for epoch in range(self.config.num_epochs):
            logging.info(f"Epoch [{epoch+1}/{self.config.num_epochs}]")

            train_loss, train_acc = self.train_one_epoch(train_loader)
            val_loss, val_acc = self.validate(test_loader)

            logging.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            logging.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

            if val_acc > best_accuracy:
                best_accuracy = val_acc
                self.save_model()
                logging.info(f"New best model saved at acc: {best_accuracy:.4f}")

        return self.model

    def save_model(self):
        self.config.model_save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), self.config.model_save_path)
        logging.info(f"Model saved to {self.config.model_save_path}")
