from omegaconf import DictConfig
from hydra.utils import instantiate
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, f1_score
import numpy as np
import pandas as pd

from atif.core import AbstractModel, AbstractDataset, Mode
from atif.logger import FileLogger


class CustomModelWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, cfg: DictConfig, mode: Mode, offset=0.5, seed=1234, n_estimators=100, 
                 softmax_tau=1.0, eps=0.0, sigma=0.1):
        self.cfg = cfg
        self.mode = mode
        self.offset = offset
        self.seed = seed
        self.n_estimators = n_estimators
        self.softmax_tau = softmax_tau
        self.eps = eps
        self.sigma = sigma
        self.model = instantiate(cfg.model.type_model, _recursive_=False)
        self.model.setup(cfg.model)

    def fit(self, X, y):
        self.classes_ = np.unique(y)  # Set the classes_ attribute
        if self.mode == Mode.CLASSIC:
            self.model.create_tree(self.seed, self.n_estimators, len(X))
            self.model.fit(X, y)
            self.model.set_param_isolation_forest(self.offset)
        else:
            self.model.create_tree(self.seed, self.n_estimators, len(X))
            self.model.fit(X, y)
            self.model.set_param_attention(self.eps, self.sigma, self.softmax_tau)
            self.model.optimize_weights(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def get_params(self, deep=True):
        return {
            'cfg': self.cfg,
            'mode': self.mode,
            'offset': self.offset,
            'seed': self.seed,
            'n_estimators': self.n_estimators,
            'softmax_tau': self.softmax_tau,
            'eps': self.eps,
            'sigma': self.sigma
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self


class Solution:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self._model: AbstractModel = instantiate(cfg.model.type_model, _recursive_=False)
        self._dataset: AbstractDataset = instantiate(cfg.dataset, _recursive_=False)
        self._logger = FileLogger()
        self._setup(cfg)
        self._report_path = cfg.report_path

    def _setup(self, cfg: DictConfig):
        self._logger.setup(cfg.logger)
        self._dataset.load()

    def run(self):
        X, y = self._dataset.X, self._dataset.y

        if self._model.get_type_model() == Mode.CLASSIC:
            param_grid = {
                'offset': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                'seed': [1234 + 7 * i for i in range(30)],
                'n_estimators': [25, 50, 100, 150, 200, 250, 300]
            }
            relevant_params = ['param_offset', 'param_seed', 'param_n_estimators']
        else:
            param_grid = {
                'softmax_tau': [0.1, 10, 20, 30, 40],
                'n_estimators': [25, 50, 100, 150, 200, 250, 300],
                'seed': [1234 + 7 * i for i in range(30)],
                'eps': [0.0, 0.25, 0.5, 0.75, 1.0],
                'sigma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            }
            relevant_params = ['param_softmax_tau', 'param_eps', 'param_sigma', 'param_seed', 'param_n_estimators']

        custom_model = CustomModelWrapper(cfg=self.cfg, mode=self._model.get_type_model())
        scorer = make_scorer(f1_score)
        grid_search = GridSearchCV(estimator=custom_model, param_grid=param_grid, cv=self._dataset.kf, scoring=scorer, n_jobs=-1)
        grid_search.fit(X, y)

        # Convert results to DataFrame and sort by F1 score
        self.results_df = pd.DataFrame(grid_search.cv_results_)
        self.results_df = self.results_df.loc[:, relevant_params + ['mean_test_score', 'std_test_score']]
        self.results_df.sort_values(by='mean_test_score', ascending=False, inplace=True)

        # Save all results to CSV
        self.results_df.to_csv(self._report_path + "optimization_results.csv", index=False)

        # Log the best result
        best_result = grid_search.best_params_
        self._logger.info(f"Best parameters: {best_result}, Max Average F1: {grid_search.best_score_:.4f}")

    def close(self):
        self._logger.end_logger("end model optimization")


def optimization(cfg: DictConfig):
    """Main function.

    Args:
        cfg (DictConfig): config structure by .yaml.
    """
    print("Optimization")
    solution_runner = Solution(cfg)
    solution_runner.run()
    solution_runner.close()