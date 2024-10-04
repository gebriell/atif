import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import preprocessing
from atif.core import AbstractDataset
from pylab import rcParams
import numpy as np

rcParams['figure.figsize'] = 14, 8

class DatasetAnomaly(AbstractDataset):
    def __init__(self, path: str, target_col: str = None):
        super().__init__()
        self._path = path
        self._target_col = target_col

    def load(self):
        data = pd.read_csv(self._path)
        
        # Encoding target column if necessary
        if self._target_col:
            label_encoder = preprocessing.LabelEncoder()
            data[self._target_col] = label_encoder.fit_transform(data[self._target_col])

        # Select features and target
        features = data[['distance_sum', 'distance_mean', 'distance_count', 'atanlen_mean', 
                         'dx_std', 'atanlen_sum', 'atanlen_std']]
        X = features
        y = data['outlier']

        # Count and plot class distribution
        count_class = pd.Series(y).value_counts()
        count_class.plot(kind='bar', rot=0)
        plt.title("Outlier Distribution")
        plt.xticks(range(2), ["Normal", "Outlier"])
        plt.xlabel("Class")
        plt.ylabel("Frequency")
        plt.savefig("/home/dev/files/data/env/Attention-based-isolation-forest/atif/anomaly_dataset.jpg")

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Assign to instance variables
        self.X_train = X_train.values
        self.X_test = X_test.values
        self.y_train = y_train.values
        self.y_test = y_test.values

    def plot_dataset(self, data, y, save_path):
        pass

    def get_name(self) -> str:
        count_normal = (self.y_train == 1).sum() + (self.y_test == 1).sum()
        count_anomaly = (self.y_train == -1).sum() + (self.y_test == -1).sum()
        return f"anomaly_normal={count_normal}_anomaly={count_anomaly}"