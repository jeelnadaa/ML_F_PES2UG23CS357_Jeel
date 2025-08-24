import numpy as np
from collections import Counter

def get_entropy_of_dataset(data: np.ndarray) -> float:
    """
    Calculate the entropy of the entire dataset using the target variable (last column).
    
    Args:
        data (np.ndarray): Dataset where the last column is the target variable
    
    Returns:
        float: Entropy value calculated using the formula: 
               Entropy = -Σ(p_i * log2(p_i)) where p_i is the probability of class i
    
    Example:
        data = np.array([[1, 0, 'yes'],
                        [1, 1, 'no'],
                        [0, 0, 'yes']])
        entropy = get_entropy_of_dataset(data)
        # Should return entropy based on target column ['yes', 'no', 'yes']
    """
    # Extract target column
    target = data[:, -1]
    # Count occurrences of each class
    values, counts = np.unique(target, return_counts=True)
    # Compute probabilities
    probabilities = counts / len(target)
    # Calculate entropy while avoiding log2(0)
    entropy = -np.sum([p * np.log2(p) for p in probabilities if p > 0])
    return entropy


def get_avg_info_of_attribute(data: np.ndarray, attribute: int) -> float:
    """
    Calculate the average information (weighted entropy) of a specific attribute.
    
    Args:
        data (np.ndarray): Dataset where the last column is the target variable
        attribute (int): Index of the attribute column to calculate average information for
    
    Returns:
        float: Average information calculated using the formula:
               Avg_Info = Σ((|S_v|/|S|) * Entropy(S_v)) 
               where S_v is subset of data with attribute value v
    """
    total_samples = len(data)
    avg_info = 0.0

    # Get unique values of the attribute
    values, counts = np.unique(data[:, attribute], return_counts=True)

    # For each value of the attribute, compute weighted entropy
    for val, count in zip(values, counts):
        subset = data[data[:, attribute] == val]
        weight = count / total_samples
        entropy_subset = get_entropy_of_dataset(subset)
        avg_info += weight * entropy_subset

    return avg_info


def get_information_gain(data: np.ndarray, attribute: int) -> float:
    """
    Calculate the Information Gain for a specific attribute.
    
    Args:
        data (np.ndarray): Dataset where the last column is the target variable
        attribute (int): Index of the attribute column to calculate information gain for
    
    Returns:
        float: Information gain calculated using the formula:
               Information_Gain = Entropy(S) - Avg_Info(attribute)
               Rounded to 4 decimal places
    """
    # Compute dataset entropy
    total_entropy = get_entropy_of_dataset(data)
    # Compute average information of the attribute
    avg_info = get_avg_info_of_attribute(data, attribute)
    # Compute information gain
    info_gain = total_entropy - avg_info
    return round(info_gain, 4)


def get_selected_attribute(data: np.ndarray) -> tuple:
    """
    Select the best attribute based on highest information gain.
    
    Args:
        data (np.ndarray): Dataset where the last column is the target variable
    
    Returns:
        tuple: A tuple containing:
            - dict: Dictionary mapping attribute indices to their information gains
            - int: Index of the attribute with the highest information gain
    """
    n_attributes = data.shape[1] - 1  # Exclude target column
    gains = {}

    # Compute information gain for each attribute
    for attr in range(n_attributes):
        gains[attr] = get_information_gain(data, attr)

    # Select attribute with maximum information gain
    best_attr = max(gains, key=gains.get)

    return gains, best_attr
