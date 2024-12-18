from typing import  List, Tuple

from joblib import load
from sklearn.linear_model import LogisticRegression


class SklearnClassifier():
    """
    A classifier that uses a pre-trained Scikit-Learn model to make predictions.

    Attributes:
        model (LogisticRegression): The pre-trained Scikit-Learn model.
    """

    def __init__(self, sklearn_model_path):
        """
        Initializes the SklearnClassifier with a pre-trained model.

        Args:
            sklearn_model_path (str): The path to the pre-trained Scikit-Learn model.
        """
        self.model: LogisticRegression = load(sklearn_model_path)

    def predict(self, input_data: List[List[float]]):
        """
        Makes predictions using the pre-trained Scikit-Learn model.

        Args:
            input_data (List[List[float]]): A list of input samples, where each sample is a list of features.

        Returns:
            List[List[float]]: A list of prediction probabilities for each input sample.
        """
        probas = self.model.predict_proba(input_data).tolist()
        return probas
