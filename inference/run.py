"""
Script loads the latest trained model, data for inference and predicts results.
Imports necessary packages and modules.
"""

import argparse
import json
import logging
import os
import sys
import time
import torch
import torch.nn as nn
import pandas as pd
from datetime import datetime
from torch.utils.data import DataLoader

# Adds the root directory to system path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(ROOT_DIR))

# Change to CONF_FILE = "settings.json" if you have problems with env variables
# I had problems, so I need to change it.
CONF_FILE = "./settings.json"

from utils import get_project_dir, configure_logging

# Loads configuration settings from JSON
with open(CONF_FILE, "r") as file:
    conf = json.load(file)

""" When build the inference Docker image, unfortunately, I had problems to import:
    from training.train import IrisClassifier
   So I need to copy, paste it from the train.py 
"""
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


# Defines paths
DATA_DIR = get_project_dir(conf['general']['data_dir'])
MODEL_DIR = get_project_dir(conf['general']['models_dir'])
RESULTS_DIR = get_project_dir(conf['general']['results_dir'])

# Initializes parser for command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--infer_file",
                    help="Specify inference data file",
                    default=conf['inference']['inp_table_name'])
parser.add_argument("--out_path",
                    help="Specify the path to the output table")


def get_latest_model_path() -> str:
    """Gets the path of the latest saved model"""
    latest = None
    for (dirpath, dirnames, filenames) in os.walk(MODEL_DIR):
        for filename in filenames:
            if not latest or datetime.strptime(latest, conf['general']['datetime_format'] + '.pth') < \
                    datetime.strptime(filename, conf['general']['datetime_format'] + '.pth'):
                latest = filename
    return os.path.join(MODEL_DIR, latest)

def get_model_by_path(path: str) -> IrisClassifier:
    """Loads and returns the specified model"""
    try:
        new_model = IrisClassifier()
        new_model.load_state_dict(torch.load(path))
        return new_model
    except Exception as e:
        logging.error(f'An error occurred while loading the model: {e}')
        sys.exit(1)


def get_inference_data(path: str) -> pd.DataFrame:
    """Loads and returns data for inference from the specified csv. file"""
    try:
        df = pd.read_csv(path)
        return df
    except Exception as e:
        logging.error(f"An error occurred while loading inference data: {e}")
        sys.exit(1)


def making_tensors(infer_data: pd.DataFrame) -> DataLoader:
    """Turn the 'infer_data' data set into Dataloader without labels."""
    tensor_data = torch.tensor(infer_data.values, dtype=torch.float32)
    test_loader = DataLoader(tensor_data, batch_size=32, shuffle=False)
    return test_loader
    

def predict_results(model: IrisClassifier, infer_data: pd.DataFrame, test_loader: DataLoader) -> pd.DataFrame:
    """The model predict the results and 'infer_data' will contains these predicted labels."""
    model.eval()
    with torch.no_grad():
        predictions = []
        for inputs in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.numpy())
    infer_data['predictions'] = predictions
    return infer_data


def store_results(results: pd.DataFrame, path: str = None) -> None:
    """Store the prediction results in 'results' directory with current datetime as a filename"""
    if not path:
        if not os.path.exists(RESULTS_DIR):
            os.makedirs(RESULTS_DIR)
        path = datetime.now().strftime(
            conf['general']['datetime_format']) + '.csv'
        path = os.path.join(RESULTS_DIR, path)
    pd.DataFrame(results).to_csv(path, index=False)
    logging.info(f'Results saved to {path}')


def main():
    """Main function"""
    configure_logging()
    args = parser.parse_args()
    model = get_model_by_path(get_latest_model_path())
    infer_file = args.infer_file
    infer_data = get_inference_data(os.path.join(DATA_DIR, infer_file))
    test_loader = making_tensors(infer_data)

    start_time = time.time()
    results = predict_results(model, infer_data, test_loader)    
    end_time = time.time()

    logging.info(f"Maked predictions in {end_time - start_time} seconds.")
    store_results(results, args.out_path)

    logging.info(f'Prediction results: \n{results}')


if __name__ == "__main__":
    main()
