import numpy as np
import torch

from paper_demo.metrics.AccuracyInterpMetricAbstract import AccuracyInterpMetricAbstract
from paper_demo.metrics.ffvus.vus_torch import VUSTorch


class AVUSI(AccuracyInterpMetricAbstract):
    """Takes an anomaly scoring and ground truth labels to compute and apply a threshold to the scoring.

    Subclasses of this abstract base class define different strategies to put a threshold over the anomaly scorings.
    All strategies produce binary labels (0 or 1; 1 for anomalous) in the form of an integer NumPy array.
    The strategy :class:`~timeeval.metrics.thresholding.NoThresholding` is a special no-op strategy that checks for
    already existing binary labels and keeps them untouched. This allows applying the metrics on existing binary
    classification results.
    """

    vusi_dict = None

    def __init__(self, slope:int=64, n_thresholds=50) -> None:
        # self.max_k: Optional[int] = max_k
        # self.buffer = buffer
        self.n_thresholds = n_thresholds
        self.slope = slope

        # def supports_continuous_scorings(self) -> bool:
        #     return True

    @property
    def name(self) -> str:
        return f'aVUSi'.upper()

    def get_name_template(self) -> str:
        return 'aVUSi'.upper()

    def get_name_with_m(self, m) -> str:
        return f'VUSi(m={m})'

    def get_vusi_dict(self):
        return self.vusi_dict



    def score(self, y_true, y_score, interp_score):
        # interp_score[y_true == 0] = 0  # Set interpretability scores of normal samples to the minimum score

        m_list = np.linspace(0,1, self.n_thresholds)
        vusi_scores = []

        interp_score_copy = interp_score.copy()
        self.vusi_dict = dict()
        for m in m_list:
            interp_score_copy[y_true == 0] = m # Set interpretability scores of normal samples to m

            new_anomaly_scores = y_score * interp_score_copy

            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            vus = VUSTorch(slope_size=self.slope, device=device)
            value_data, timing = vus.compute(torch.from_numpy(y_true).to(device),
                                             torch.from_numpy(np.round(new_anomaly_scores, decimals=2)).to(device))
            vusi_scores.append(value_data[0].item())
            self.vusi_dict[m] = value_data[0].item()

        return np.trapz(vusi_scores, m_list)

    # def score_for_different_k(self, y_true_univariate, y_score_univariate, y_true_multivariate: np.ndarray, dimension_contribution_multivariate: np.ndarray) -> Dict:
    #     assert y_true_multivariate.ndim == 2
    #     assert dimension_contribution_multivariate.ndim == 2
    #     assert y_true_univariate.ndim == 1
    #     assert y_score_univariate.ndim == 1
    #     # fpr, tpr, thresholds = roc_curve(y_true.reshape(-1), y_score.reshape(-1))
    #     # result = auc(fpr, tpr)
    #
    #     y_true = (y_true_multivariate.sum(axis=1) >= 1).astype(float)
    #     assert (y_true == y_true_univariate).all()
    #
    #     # y_score_univariate_sorted = -np.sort(-y_score_univariate)
    #     #
    #     # thresholds = []
    #     # precision = []
    #     # recall = []
    #     # pred_for_thresholds = []  # Store predictions for each threshold
    #     #
    #     # for k, i in enumerate(np.linspace(0, len(y_score_univariate) - 1, self.n_thresholds).astype(int)):
    #     #     threshold = y_score_univariate_sorted[i]
    #     #     pred = (y_score_univariate >= threshold).astype(float)
    #     #     tp = np.sum((pred == 1) & (y_true_univariate == 1))
    #     #     fp = np.sum((pred == 1) & (y_true_univariate == 0))
    #     #     fn = np.sum((pred == 0) & (y_true_univariate == 1))
    #     #     precision.append(tp / (tp + fp) if (tp + fp) > 0 else 0)
    #     #     recall.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
    #     #     thresholds.append(threshold)
    #     #     pred_for_thresholds.append(pred)
    #     # thresholds = np.array(thresholds)
    #     # precision = np.array(precision)
    #     # recall = np.array(recall)
    #     # pred_for_thresholds = np.stack(pred_for_thresholds, axis=1)
    #
    #     estimated_dimension_contribution = estimate_dimension_contribution_with_a_buffer(
    #         dimension_contribution_multivariate,
    #         buffer=self.buffer)
    #     estimated_dimension_contribution = estimated_dimension_contribution / estimated_dimension_contribution.sum(
    #         axis=1, keepdims=True)
    #     # dimension_contribution_ranking = np.argsort(estimated_dimension_contribution, axis=1)
    #
    #     interpretability_scores = _ndcg_sample_scores(y_true_multivariate, estimated_dimension_contribution, k=self.max_k)
    #     interpretability_scores[y_true_univariate == 0] = 1.0  # Set interpretability scores of normal samples to the minimum score
    #     new_anomaly_scores = y_score_univariate * interpretability_scores
    #
    #     device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #     vus = VUSTorch(slope_size=self.slope, device=device)
    #     value_data, timing = vus.compute(torch.from_numpy(y_true).to(device), torch.from_numpy(np.round(new_anomaly_scores, decimals=2)).to(device))
    #     result_dict = {self.max_k: value_data[0].item()}
    #     return result_dict