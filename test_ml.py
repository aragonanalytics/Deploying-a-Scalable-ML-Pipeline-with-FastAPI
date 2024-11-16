import pytest
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from ml.model import train_model, compute_model_metrics
# TODO: add necessary import

# TODO: implement the first test. Change the function name and input as needed
def test_data_loading():
    """
    # Test if the data is loaded correctly
    """
    # Your code here
    data = pd.read_csv('data/census.csv')
    # Check if data is a dataframe
    assert isinstance(data, pd.DataFrame), "Data should be in a DataFrame"
    # Check if data is present, there are no columns or rows that = 0 
    assert data.shape[0] > 0, "There should be more than 0 rows"
    assert data.shape[1] > 0, "There should be more than 0 columns"
    pass


# TODO: implement the second test. Change the function name and input as needed
def test_train_test_split():
    """
    # Test if the train and test datasets have the expected 80/20 split
    """
    # Your code here
    data = pd.read_csv('data/census.csv')
    train, test = train_test_split(data, test_size = 0.2, random_state = 42)
    # Check if the train dataset is 80% of the total data
    assert len(train) == int(0.8 * len(data)), "Train dataset size should be 80% of the total data"
    # Check if the test dataset is 20% of the total data
    assert abs(len(test) - int(0.2 * len(data))) <= 1, "Test dataset size should be 20% of the total data"
    # Check if the datasets are DataFrames
    assert isinstance(train, pd.DataFrame), "Train dataset should be a DataFrame"
    assert isinstance(test, pd.DataFrame), "Test dataset should be a DataFrame"
    pass


# TODO: implement the third test. Change the function name and input as needed
def test_compute_model_metrics():
    """
    # Test if prediction, recall, and F1 score are correctly calculated
    """
    # Your code here
    # Sample y_true and y_pred
    y_true = [0, 1, 0, 1, 0, 1]
    y_pred = [0, 1, 0, 1, 1, 0]
    
    # Compute metrics
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    # Check if the metrics are within the valid range
    assert 0 <= precision <= 1, "Precision should be between 0 and 1"
    assert 0 <= recall <= 1, "Recall should be between 0 and 1"
    assert 0 <= f1 <= 1, "F1 score should be between 0 and 1"
    pass

if __name__ == "__main__":
    pytest.main()