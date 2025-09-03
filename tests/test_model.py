import unittest
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Paths
MODEL_PATH = "model/iris_model.pkl"
CSV_PATH = "iris.csv"


class TestModelTraining(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load model and dataset once for all tests
        cls.model = joblib.load(MODEL_PATH)
        cls.iris_data = pd.read_csv(CSV_PATH)

    def test_model_instance(self):
        """Check if the loaded model is a RandomForestClassifier"""
        self.assertIsInstance(self.model, RandomForestClassifier)

    def test_feature_importances(self):
        """Ensure model has feature importances with at least 4 features"""
        self.assertGreaterEqual(len(self.model.feature_importances_), 4)

    def test_model_prediction(self):
        """Check model prediction shape using first row of data"""
        features = self.iris_data.iloc[0, :-1].values.reshape(1, -1)
        pred = self.model.predict(features)
        self.assertEqual(pred.shape, (1,))


if __name__ == "__main__":
    unittest.main()




