import unittest
import pandas as pd
import os
import sys
import json
import torch
from torch.utils.data import TensorDataset, DataLoader

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(ROOT_DIR))
CONF_FILE = "./settings.json"

from training.train import DataProcessor, Training

class TestDataProcessor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(CONF_FILE, "r") as file:
            conf = json.load(file)
        cls.data_dir = conf['general']['data_dir']
        cls.train_path = os.path.join(cls.data_dir, conf['train']['table_name'])

    def test_data_extraction(self):
        dp = DataProcessor()
        df = dp.data_extraction(self.train_path)
        self.assertIsInstance(df, pd.DataFrame)

    def test_prepare_data(self):
        dp = DataProcessor()
        df = dp.prepare_data(100)
        self.assertEqual(df.shape[0], 100)


class TestTraining(unittest.TestCase):
    def test_train(self):
        tr = Training()

        X_train = pd.DataFrame({
            'x1': [-1.51, -0.17, 1.04, -1.26],
            'x2': [1.25, 3.09, 0.1, 0.79],
            'x3': [-1.57, -1.28, 0.36, -1.23],
            'x4': [-1.32, -1.05, 0.26, -1.32],

        })
        y_train = pd.Series([0, 0, 1, 0])
        X_tensor = torch.tensor(X_train.values, dtype=torch.float32)
        y_tensor = torch.tensor(y_train.values, dtype=torch.long)

        # Create a TensorDataset
        dataset = TensorDataset(X_tensor, y_tensor)

        # Define batch size
        batch_size = 32  # You can adjust this according to your needs

        # Create DataLoader
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        tr.train(train_loader)

if __name__ == '__main__':
    unittest.main()
    