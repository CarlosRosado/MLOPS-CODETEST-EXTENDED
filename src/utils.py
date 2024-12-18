import numpy as np
import ast

def load_labels(file_path):
    """
    Load labels from a file.

    Args:
        file_path (str): Path to the file containing the labels.

    Returns:
        list: A list of labels.
    """
    with open(file_path, 'r') as file:
        labels = ast.literal_eval(file.read().strip())
    return labels

def validate_input(data):
    """
    Validate the input data.

    Args:
        data (dict): The input data containing 'crystalData'.

    Returns:
        bool: True if the input data is valid, False otherwise.
    """
    if 'crystalData' not in data:
        return False
    if not isinstance(data['crystalData'], list):
        return False
    for sample in data['crystalData']:
        if not isinstance(sample, list) or len(sample) != 4:
            return False
    return True

def format_response(predictions, labels):
    """
    Format the response with predictions and scores.

    Args:
        predictions (numpy.ndarray): The predictions from the model.

    Returns:
        dict: A dictionary containing the formatted response.
    """
    #labels = ["blue", "green", "yellow"]
    
    response = {
        "prediction": [],
        "scores": []
    }
    for prediction in predictions:
        label_index = np.argmax(prediction)
        response["prediction"].append(labels[label_index])
        response["scores"].append({
            labels[0]: prediction[0],
            labels[1]: prediction[1],
            labels[2]: prediction[2]
        })
    return response