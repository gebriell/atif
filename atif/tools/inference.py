from omegaconf import DictConfig
from hydra.utils import instantiate

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_curve, auc, precision_score, recall_score

from atif.core import AbstractModel, AbstractDataset, Mode
from atif.logger import FileLogger

class Solution:
    def __init__(self, cfg: DictConfig):
        self._model: AbstractModel = instantiate(cfg.model.type_model, _recursive_=False)
        self._dataset: AbstractDataset = instantiate(cfg.dataset, _recursive_=False)
        self._logger = FileLogger()
        self._setup(cfg)

    def _setup(self, cfg: DictConfig):
        self._logger.setup(cfg.logger)
        self._dataset.load()
        self._model.setup(cfg.model)

    def run(self, cfg: DictConfig):
        self._logger.start_logger("model: " + self._model.__str__())
        
        # Get the splits for k-fold
        for fold_idx, (train_index, test_index) in enumerate(self._dataset.get_kfold_splits()):
            print(f"Processing fold {fold_idx + 1}...")
            X_train, X_test = self._dataset.X[train_index], self._dataset.X[test_index]
            y_train, y_test = self._dataset.y[train_index], self._dataset.y[test_index]

            self._model.create_tree(cfg.model.seed, cfg.model.n_estimators, len(X_train))
            self._model.fit(X_train, y_train)
            
            if self._model.get_type_model() == Mode.ATTENTION:
                self._model.optimize_weights(X_train, y_train)
            
            y_prediction = self._model.predict(X_test, y_test)
            self._plot_dataset(y_prediction, X_train, y_train, X_test, y_test)
            self._log_metrics(y_prediction, y_test)

    def _plot_dataset(self, y_prediction, X_train, y_train, X_test, y_test):
        self._dataset.plot_dataset(X_train, y_train,
                                   self._logger.directory_image + "/train_"
                                   + self._dataset.get_name() + ".jpg")
        self._dataset.plot_dataset(X_test, y_test,
                                   self._logger.directory_image + "/test_"
                                   + self._dataset.get_name() + ".jpg")
        self._dataset.plot_dataset(X_test, y_prediction,
                                   self._logger.directory_image + "/predict_"
                                   + self._model.get_name() + "_" + self._dataset.get_name() + ".jpg")

    def _log_metrics(self, y_prediction, y_test):
        confusion = confusion_matrix(y_test, y_prediction)
        TN, FP, FN, TP = confusion.flat
        TPR = TP / (TP + FN)
        FPR = FP / (FP + TN)
        normal = y_test[y_test == 0]
        anomalies = y_test[y_test == 1]
        F1 = f1_score(y_test, y_prediction)
        precision = precision_score(y_test, y_prediction)
        recall = recall_score(y_test, y_prediction)
        self._logger.info(
            f"Proportion anomalies/normal = {len(anomalies)}/{len(normal)} = {(len(anomalies) / len(normal)) * 100:.1f}%")
        self._logger.info(f"F1 score {F1:.4f}")
        self._logger.info(f"Precision score {precision:.4f}")
        self._logger.info(f"Recall score {recall:.4f}")
        self._logger.info(f"TPR score {TPR:.4f}, FPR {FPR:.4f}")
        self._logger.info(f"TP {TP:.4f}, FP {FP:.4f}")
        self._logger.info(f"FN {FN:.4f}, TN {TN:.4f}")
        fpr, tpr, thresholds = roc_curve(y_test, y_prediction, pos_label=1)
        self._logger.info(f"auc {auc(fpr, tpr):.4f}")
        self._logger.info(f"accuracy {accuracy_score(y_prediction, y_test):.4f}")

    def close(self):
        self._logger.end_logger("end model inference")


def inference(cfg: DictConfig):
    """Main function.

    Args:
        cfg (DictConfig): config structure by .yaml.
    """
    print("Inference")
    solution_runner = Solution(cfg)
    solution_runner.run(cfg)
    solution_runner.close()
