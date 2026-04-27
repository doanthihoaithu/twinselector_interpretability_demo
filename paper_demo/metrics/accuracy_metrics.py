import numpy as np
import torch

from paper_demo.metrics.AccuracyMetricAbstract import AccuracyMetricAbstract
from paper_demo.metrics.ffvus.vus_torch import VUSTorch


class PrAUC(AccuracyMetricAbstract):

    @property
    def name(self) -> str:
        return "PR_AUC"

    def score(self, y_true, y_score) -> float:
        from sklearn.metrics import auc, precision_recall_curve

        precision, recall, _ = precision_recall_curve(y_true, y_score)
        return auc(recall, precision)

class RocAUC(AccuracyMetricAbstract):

    @property
    def name(self) -> str:
        return "ROC_AUC"

    def score(self, y_true, y_score) -> float:
        from sklearn.metrics import roc_auc_score

        return roc_auc_score(y_true, y_score)




class Precision(AccuracyMetricAbstract):

    @property
    def name(self) -> str:
        return "Precision"

    def score(self, y_true, y_score) -> float:
        from sklearn.metrics import precision_score

        return precision_score(y_true, y_score)

class Recall(AccuracyMetricAbstract):

    @property
    def name(self) -> str:
        return "Recall"

    def score(self, y_true, y_score) -> float:
        from sklearn.metrics import recall_score

        return recall_score(y_true, y_score)

class F1Score(AccuracyMetricAbstract):

    @property
    def name(self) -> str:
        return "F1"

    def score(self, y_true, y_score) -> float:
        from sklearn.metrics import f1_score

        return f1_score(y_true, y_score)

class VUSPR(AccuracyMetricAbstract):

    slope: int = 10

    def __init__(self, slope: int = 10):
        super().__init__()
        self.slope = slope

    @property
    def name(self) -> str:
        return "VUS_PR"

    def score(self, y_true, y_score) -> float:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        vus = VUSTorch(slope_size=self.slope, device=device)
        value_data, timing = vus.compute(torch.from_numpy(y_true).to(device),
                                         torch.from_numpy(np.round(y_score, decimals=2)).to(device))
        return value_data[0].item()
