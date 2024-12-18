from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """
    A neural network model for classification using PyTorch.

    Attributes:
        layer1 (nn.Linear): The first linear layer.
        layer2 (nn.Linear): The second linear layer.
        layer3 (nn.Linear): The third linear layer.
    """
    def __init__(self, input_dim=4):
        """
        Initializes the Model with the specified input dimension.

        Args:
            input_dim (int): The number of input features. Default is 4.
        """
        super(Model, self).__init__()
        self.layer1 = nn.Linear(input_dim, 50)
        self.layer2 = nn.Linear(50, 50)
        self.layer3 = nn.Linear(50, 3)

    def forward(self, x):
        """
        Defines the forward pass of the model.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor with prediction probabilities.
        """
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.softmax(self.layer3(x), dim=1)
        return x

class PytorchClassifier:
    """
    A classifier that uses a pre-trained PyTorch model to make predictions.

    Attributes:
        model (Model): The pre-trained PyTorch model.
    """
    def __init__(self, pytorch_model_path):
        """
        Initializes the PytorchClassifier with a pre-trained model.

        Args:
            pytorch_model_path (str): The path to the pre-trained PyTorch model.
        """
        self.model = Model()
        self.model.load_state_dict(torch.load(pytorch_model_path))
        self.model.eval()

    def predict(self, input_data: List[List[float]]):
        """
        Makes predictions using the pre-trained PyTorch model.

        Args:
            input_data (List[List[float]]): A list of input samples, where each sample is a list of features.

        Returns:
            List[List[float]]: A list of prediction probabilities for each input sample.
        """
        with torch.no_grad():
            probas = self.model(torch.Tensor(input_data)).tolist()
        return probas
