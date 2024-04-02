import datetime
import sys

import torch.nn
from torch.optim import Adam
from tqdm import tqdm
from transformers import BertTokenizer

from amazon_reviews_dataset import AmazonReviewsDataset
from torch.utils.data import DataLoader
import logging as log
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
import json

from ml_models.sentiment_model import SentimentModel, transformer_name
from strong_types.ml_mode import MlMode
from strong_types.review_item import ReviewItem


class MlPipe:
    def __init__(self):
        self.logger = log.getLogger("MlPipe")
        self.logger.warning("pipe init")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        dataset_train = AmazonReviewsDataset(MlMode.TRAIN)
        dataset_test = AmazonReviewsDataset(MlMode.TEST)
        dataset_val = AmazonReviewsDataset(MlMode.VALIDATION)

        self.dataloader_train = DataLoader[ReviewItem](dataset_train, batch_size=dataset_train.batch_size,
                                                       shuffle=False)
        self.dataloader_test = DataLoader[ReviewItem](dataset_test, batch_size=dataset_test.batch_size,
                                                      shuffle=False)
        self.dataloader_val = DataLoader[ReviewItem](dataset_val, batch_size=dataset_val.batch_size,
                                                     shuffle=False)
        self.model = SentimentModel(6).to(self.device)
        self.tokenizer = BertTokenizer.from_pretrained(transformer_name)
        self.optimizer = Adam(self.model.parameters(), lr=5e-5)
        self.criterion = torch.nn.CrossEntropyLoss()

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        total_count = 0
        progress_bar = tqdm(self.dataloader_train, desc="Training", leave=True, file=sys.stdout)
        for batch in progress_bar:
            total_count += self.dataloader_train.batch_size
            self.optimizer.zero_grad()
            inputs = self.tokenizer(batch.reviewText, padding=True, truncation=True, return_tensors="pt",
                                    max_length=512).to(self.device)
            labels = batch.overall.to(self.device).long()
            outputs = self.model(**inputs)
            loss = self.criterion(outputs, labels)
            total_loss += loss.item()
            loss.backward()
            self.optimizer.step()
        avg_loss = total_loss / total_count
        return avg_loss

    def evaluate(self):
        self.model.eval()
        total_loss = 0
        total_count = 0
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch in self.dataloader_val:
                total_count += self.dataloader_train.batch_size
                inputs = self.tokenizer(batch.reviewText, padding=True, truncation=True, return_tensors="pt",
                                        max_length=512).to(self.device)
                labels = batch.overall.to(self.device).long()
                outputs = self.model(**inputs)

                probabilities = torch.softmax(outputs, dim=1)
                predictions = torch.argmax(probabilities, dim=1)

                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
        avg_loss = total_loss / total_count
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions,
                                                                   average='weighted', zero_division=0)

        return {"avg_loss": avg_loss,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1}

    def train(self, epochs=3):
        for epoch in range(epochs):
            try:
                self.logger.info("Epoch %s/%s", epoch + 1, epochs)
                self.train_epoch()
                metrics = self.evaluate()
                self.logger.warning("Metrics: %s", metrics)
                self.save_model(f"./data/model_{datetime.date.today()}_epoch_{epoch + 1}.ptn")
                self.save_metric(f"./data/model_{datetime.date.today()}_epoch_{epoch + 1}.json", metrics)
            except BaseException:
                self.save_model(f"./data/interrupted_{datetime.date.today()}_epoch{epoch + 1}.ptn")
                raise

    def save_metric(self, save_path: str, metric: dict):
        with open(save_path, 'w') as json_file:
            json.dump(metric, json_file, indent=4)

    def save_model(self, save_path: str):
        torch.save(self.model.state_dict(), save_path)

    def load_model(self, path_model: str):
        state_dict = torch.load(path_model)
        self.model.load_state_dict(state_dict)

    def predict_review(self, review_text: str) -> int:
        self.model.eval()

        inputs = self.tokenizer(review_text, padding=True, truncation=True, return_tensors="pt", max_length=512).to(
            self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

            probabilities = torch.softmax(outputs, dim=1)

            predicted_rating = torch.argmax(probabilities, dim=1).item()

        return predicted_rating
