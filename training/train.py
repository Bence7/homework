"""
This script prepares the data, runs the training, and saves the model.
"""
import argparse
from copy import deepcopy
import os
import sys
import json
import logging
import pandas as pd
import time
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, TensorDataset, random_split
from datetime import datetime
from sklearn.metrics import accuracy_score


# Comment this lines if you have problems with MLFlow installation
import mlflow
mlflow.autolog()

# Adds the root directory to system path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(ROOT_DIR))

# Change to CONF_FILE = "settings.json" if you have problems with env variables
CONF_FILE = './settings.json'
from utils import get_project_dir, configure_logging

# Loads configuration settings from JSON
with open(CONF_FILE, "r") as file:
    conf = json.load(file)

# Defines paths
DATA_DIR = get_project_dir(conf['general']['data_dir'])
MODEL_DIR = get_project_dir(conf['general']['models_dir'])
TRAIN_PATH = os.path.join(DATA_DIR, conf['train']['table_name'])

# Initializes parser for command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--train_file",
                    help="Specify inference data file",
                    default=conf['train']['table_name'])
parser.add_argument("--model_path",
                    help="Specify the path for the output model")


class DataProcessor():
    "Preprocessing the data"
    def __init__(self) -> None:
        pass

    def prepare_data(self, max_rows: int = None) -> pd.DataFrame:
        logging.info("Preparing data for training...")
        df = self.data_extraction(TRAIN_PATH)
        df = self.data_rand_sampling(df, max_rows)
        return df

    def data_extraction(self, path: str) -> pd.DataFrame:
        logging.info(f"Loading data from {path}...")
        return pd.read_csv(path)

    def data_rand_sampling(self, df: pd.DataFrame, max_rows: int) -> pd.DataFrame:
        if not max_rows or max_rows < 0:
            logging.info('Max_rows not defined. Skipping sampling.')
        elif len(df) < max_rows:
            logging.info(
                'Size of dataframe is less than max_rows. Skipping sampling.')
        else:
            df = df.sample(n=max_rows, replace=False,
                           random_state=conf['general']['random_state'])
            logging.info(f'Random sampling performed. Sample size: {max_rows}')
        return df


class IrisClassifier(nn.Module):
    """Class of the Multiclassification Neural Network model.
    It has 4 input features, and will predict 1 target label from 3."""
    def __init__(self):
        super(IrisClassifier, self).__init__()
        self.fc1 = nn.Linear(4, 8)
        self.fc2 = nn.Linear(8, 3)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x


class Training():
    """Class of the training:
    It contains:
        0) run_training:
            1) making_tensors
            2) data_split
            3) create_dataloaders
            4) train
            5) evaluating
            6) save
    """

    def __init__(self) -> None:
        self.model = IrisClassifier()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=conf['train']['learning_rate'])


    def run_training(self, df: pd.DataFrame, out_path: str = None) -> None:
        """Training pipeline
            'df' parameter: the model will be trained on this data set
            'out_path' parameter: the model will be saved on this path
        """
        logging.info("Running training...")
        dataset = self.making_tensors(df)
        train_dataset, test_dataset = self.data_split(dataset=dataset)
        train_loader, test_loader = self.create_dataloaders(train_dataset=train_dataset, test_dataset=test_dataset)
        start_time = time.time()
        self.train(train_loader=train_loader)
        end_time = time.time()
        logging.info(f"Training completed in {end_time - start_time} seconds.")
        self.evaluating(test_loader=test_loader)
        self.save(out_path)

    def making_tensors(self, df: pd.DataFrame) -> TensorDataset:
        """Making a TensorDataset from the 'df' data set"""
        logging.info("Making tensors...")
        X = df.drop(columns=['label'], axis=1).values
        y = df['label']
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(y)
        dataset = TensorDataset(X_tensor, y_tensor)
        return dataset

    def data_split(self, dataset: TensorDataset):
        logging.info("Splitting data into training and test sets...")
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(
            dataset, [train_size, test_size])
        return train_dataset, test_dataset

    def create_dataloaders(self, train_dataset, test_dataset):
        logging.info("Creating train and test dataloaders...")
        batch_size = conf['train']['batch_size']
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, test_loader

    def train(self, train_loader: DataLoader) -> None:
        logging.info("Training the model...")
        self.model.train()
        epochs = conf['train']['n_epochs']
        for epoch in range(epochs):
            for inputs, labels in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()        

    def evaluating(self, test_loader: DataLoader):
        logging.info("Trained model making predictions...")
        self.model.eval()
        with torch.no_grad():
            predictions = []
            true_labels = []
            for inputs, labels in test_loader:
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)
                predictions.extend(predicted.numpy())
                true_labels.extend(labels.numpy())
        accuracy = accuracy_score(true_labels, predictions)
        logging.info(f"accuraccy_score: {accuracy}")
        return predictions

    def save(self, path: str) -> None:
        logging.info("Saving the model...")
        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)

        if not path:
            path = os.path.join(MODEL_DIR, datetime.now().strftime(
                conf['general']['datetime_format']) + '.pth')
        else:
            path = os.path.join(MODEL_DIR, path)

        torch.save(self.model.state_dict(), path)


def main():
    configure_logging()

    data_proc = DataProcessor()
    tr = Training()

    df = data_proc.prepare_data(max_rows=conf['train']['data_sample'])
    tr.run_training(df)


if __name__ == "__main__":
    main()
