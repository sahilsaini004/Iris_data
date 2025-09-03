



import pandas as pd

import unittest
import joblib
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

data_path = 'model/iris_model.pkl'
csv_path = 'data/iris.csv'


class TestModelTraining(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = joblib.load(data_path)
        cls.iris_data = pd.read_csv(csv_path)

    def test_model_instance(self):
        self.assertIsInstance(self.model, RandomForestClassifier)

    def test_feature_importances(self):
        self.assertGreaterEqual(len(self.model.feature_importances_), 4)

    def test_model_prediction(self):
        # Use the first row of iris.csv for prediction
        features = self.iris_data.iloc[0, :-1].values.reshape(1, -1)
        pred = self.model.predict(features)
        self.assertEqual(pred.shape, (1,))

if __name__ == '__main__':
    unittest.main()
