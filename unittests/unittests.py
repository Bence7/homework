import unittest
import numpy as np
import pandas as pd
import os
import sys
import json
from sklearn import logger
import torch
from torch.utils.data import TensorDataset, DataLoader

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(ROOT_DIR))
CONF_FILE = "./settings.json"

from data_process.data_generation import IrisDatasetGenerator
from training.train import DataProcessor, Training

class TestDataProcessor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(CONF_FILE, "r") as file:
            conf = json.load(file)
        cls.data_dir = conf['general']['data_dir']
        cls.train_path = os.path.join(
            cls.data_dir, conf['train']['table_name'])

    def test_data_extraction(self):
        dp = DataProcessor()
        df = dp.data_extraction(self.train_path)
        self.assertIsInstance(df, pd.DataFrame)

    def test_prepare_data(self):
        dp = DataProcessor()
        df = dp.prepare_data(100)
        self.assertEqual(df.shape[0], 100)


class TestDataGenerator(unittest.TestCase):
    def test_read_dataset(self):
        dg = IrisDatasetGenerator()
        train_df, inference_df = dg.read_dataset()
        self.assertEqual(train_df.shape, (120, 5))
        self.assertEqual(inference_df.shape, (30, 4))

    def test_splitting(self):
        dg = IrisDatasetGenerator()
        _, test_df=dg.splitting_dataset(np.ones((5, 4)), [1,1,1,1,1])
        expected_values = np.array([[0, 0, 0, 0]])

        self.assertEqual(np.array_equal(test_df.values, expected_values), True)

    def test_scaling(self):  
        dg = IrisDatasetGenerator()
        scaled_df= pd.DataFrame(dg.scaling(pd.DataFrame([0.3, 3.1, 3.5])))
        result_df = pd.DataFrame([[-1.40487872], [0.56195149], [0.84292723]])
        self.assertEqual(np.array(scaled_df.values).any(), np.array(result_df.values).any())

class TestTraining(unittest.TestCase):
    def setUp(self):
        self.trained_model = self.test_train()

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

        dataset = TensorDataset(X_tensor, y_tensor)

        batch_size = 32

        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        tr.train(train_loader)

        self.assertIsNotNone(tr.model.state_dict())
        return tr.model
    
    def test_evaluating(self):
        tr = Training()
        tr.model = self.trained_model
        data = [
            [-1.51, 1.25, -1.57, -1.32],
            [-0.17, 3.09, -1.28, -1.05],
            [0.55,-1.28, 0.71, 0.92],
            [0.67, 0.33, 0.42, 0.40]
        ]
        target = [[0],[0],[2],[1]]

        X_data = torch.tensor(data, dtype=torch.float32)
        y_data = torch.tensor(target, dtype=torch.long)
        dataset = TensorDataset(X_data, y_data)

        batch_size = 2 

        test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        preds=tr.evaluating(test_loader)
        logger.info(preds)
        self.assertIsNotNone(preds)


if __name__ == '__main__':
    unittest.main()
