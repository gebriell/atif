import pandas as pd
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from atif.core import AbstractDataset
from pylab import rcParams
import numpy as np

rcParams['figure.figsize'] = 14, 8

class DatasetAnomaly(AbstractDataset):
    def __init__(self, path: str, target_col: str = None, n_splits: int = 5):
        super().__init__()
        self._path = path
        self._target_col = target_col
        self.n_splits = n_splits
        self.X = None
        self.y = None
        self.kf = None

    def load(self):
        data = pd.read_csv(self._path)

        # Select features and target
        features = data[['distance_sum', 
                 'distance_mean', 
                 'distance_std', 
                 'distance_min', 
                 'distance_max', 
                 'distance_count', 
                 'atanlen_sum', 
                 'atanlen_std', 
                 'atanlen_mean', 
                 'atanlen_min', 
                 'atanlen_max', 
                 'dx_sum', 
                 'dx_mean', 
                 'dx_std', 
                 'dx_min', 
                 'dx_max', 
                 'dy_sum', 
                 'dy_mean', 
                 'dy_std', 
                 'rlen_sum', 
                 'rlen_mean', 
                 'rlen_std', 
                 'curvature_sum', 
                 'curvature_mean', 
                 'curvature_std', 
                 'curvature_min', 
                 'curvature_max', 
                 'total_path_sum', 
                 'total_path_mean', 
                 'total_path_max', 
                 'total_path_min', 
                 'straigth_line_std', 
                 'straigth_line_mean', 
                 'straigth_line_sum', 
                 'straigth_line_min', 
                 'straigth_line_max', 
                 'sinuosity_std', 
                 'sinuosity_mean', 
                 'sinuosity_sum', 
                 'sinuosity_min', 
                 'sinuosity_max', 
                 'distance_sum_zscore', 
                 'distance_mean_zscore', 
                 'atanlen_mean_zscore', 
                 'dx_mean_zscore', 
                 'dy_mean_zscore', 
                 'curvature_mean_zscore', 
                 'curvature_max_zscore', 
                 'total_path_sum_zscore', 
                 'total_path_max_zscore', 
                 'sinuosity_max_zscore']]

        self.X = features.values
        self.y = data['outlier'].values

        # Count and plot class distribution
        count_class = pd.Series(self.y).value_counts()
        count_class.plot(kind='bar', rot=0)
        plt.title("Outlier Distribution")
        plt.xticks(range(2), ["Normal", "Outlier"])
        plt.xlabel("Class")
        plt.ylabel("Frequency")
        plt.savefig("/home/dev/files/data/env/Attention-based-isolation-forest/atif/anomaly_dataset.jpg")

        # Initialize the StratifiedKFold
        self.kf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)

    def get_kfold_splits(self):
        return self.kf.split(self.X, self.y)

    def plot_dataset(self, data, y, save_path):
        pass

    def get_name(self) -> str:
        count_normal = (self.y == 1).sum()
        count_anomaly = (self.y == 0).sum()
        return f"anomaly_normal={count_normal}_anomaly={count_anomaly}"
