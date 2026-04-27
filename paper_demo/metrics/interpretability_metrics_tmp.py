# import time
# import warnings
# from typing import Optional, Tuple, List, Dict
#
# import numpy as np
# import pandas as pd
# import torch
# from sklearn.metrics import precision_recall_curve, ndcg_score
# from sklearn.metrics._ranking import _ndcg_sample_scores
#
# from metrics.Metric import Metric
# from metrics.ffvus.vus_torch import VUSTorch
#
#
# class InterpretabilityHitKScore(Metric):
#     """Takes an anomaly scoring and ground truth labels to compute and apply a threshold to the scoring.
#
#     Subclasses of this abstract base class define different strategies to put a threshold over the anomaly scorings.
#     All strategies produce binary labels (0 or 1; 1 for anomalous) in the form of an integer NumPy array.
#     The strategy :class:`~timeeval.metrics.thresholding.NoThresholding` is a special no-op strategy that checks for
#     already existing binary labels and keeps them untouched. This allows applying the metrics on existing binary
#     classification results.
#     """
#
#     def __init__(self, top_k) -> None:
#         self.top_k: Optional[int] = top_k
#
#     def __call__(self, y_true_multivariate: np.ndarray, contribution_per_var: np.ndarray):
#         return self.score(y_true_multivariate, contribution_per_var)
#     def score(self, y_true_multivariate: np.ndarray, contribution_per_var: np.ndarray) -> None:
#         if y_true_multivariate.ndim == 2 and contribution_per_var.ndim == 2:
#             y_true = (y_true_multivariate.sum(axis=1) >= 1).astype(float)
#             # y_score_per_var[np.isnan(y_score_per_var)] = 0.0
#             # y_score_per_var = convert_raw_anomaly_score_per_var_to_contribution_percentage(y_score_per_var)
#             anomaly_scores_per_var_ranking = np.argsort(contribution_per_var, axis=1)
#             top_k_anomalous_dimension = anomaly_scores_per_var_ranking[:, -self.top_k:]
#             interpretability_list = []
#             for labels, top_k_index, score_per_var in zip(y_true_multivariate, top_k_anomalous_dimension, contribution_per_var):
#                 if labels.sum() != 0.0 and not np.isnan(score_per_var).all():
#                     interpretability = labels[top_k_index].sum() / labels.sum()
#                     interpretability_list.append(interpretability)
#                 else:
#                     interpretability_list.append(np.nan)
#             # interpretability_scores = np.sqrt(np.power(multivariate_labels-anomaly_scores_per_var,2).sum(axis=1))
#             interpretability_scores = np.array(interpretability_list)
#
#             return interpretability_scores[y_true == 1].mean()
#         else:
#             return None
#         # assert y_true_multivariate.ndim == 2
#         # assert y_score_per_var.ndim == 2
#         # fpr, tpr, thresholds = roc_curve(y_true.reshape(-1), y_score.reshape(-1))
#         # result = auc(fpr, tpr)
#
#
#     # def supports_continuous_scorings(self) -> bool:
#     #     return True
#     @property
#     def name(self) -> str:
#         return f'Interpretability_Hit_{self.top_k}_Score'.upper()
#
# class InterpretabilityUnconditionalNDCGHitKScore:
#     """Takes an anomaly scoring and ground truth labels to compute and apply a threshold to the scoring.
#
#     Subclasses of this abstract base class define different strategies to put a threshold over the anomaly scorings.
#     All strategies produce binary labels (0 or 1; 1 for anomalous) in the form of an integer NumPy array.
#     The strategy :class:`~timeeval.metrics.thresholding.NoThresholding` is a special no-op strategy that checks for
#     already existing binary labels and keeps them untouched. This allows applying the metrics on existing binary
#     classification results.
#     """
#
#     def __init__(self, max_k, buffer=10) -> None:
#         self.max_k: Optional[int] = max_k
#         self.buffer = buffer
#
#     def score_for_different_k(self, y_true_multivariate: np.ndarray, dimension_contribution: np.ndarray) -> Dict:
#         assert y_true_multivariate.ndim == 2
#         assert dimension_contribution.ndim == 2
#
#         results_dict = dict() # Save results for all k
#
#         y_true = (y_true_multivariate.sum(axis=1) >= 1).astype(float)
#         # y_score_per_var[np.isnan(y_score_per_var)] = 0.0
#         # y_score_per_var = convert_raw_anomaly_score_per_var_to_contribution_percentage(y_score_per_var)
#         estimated_dimension_contribution = estimate_dimension_contribution_with_a_buffer(dimension_contribution, buffer=self.buffer)
#         estimated_dimension_contribution = estimated_dimension_contribution/estimated_dimension_contribution.sum(axis=1, keepdims=True)
#         dimension_contribution_ranking = np.argsort(estimated_dimension_contribution, axis=1)
#
#         for k in range(1, self.max_k + 1):
#             interpretability_score = ndcg_score(y_true_multivariate[y_true == 1.0], estimated_dimension_contribution[y_true == 1.0], k=k)
#             results_dict[k] = interpretability_score
#             # for labels, top_k_index, contribution_per_var in zip(y_true_multivariate, top_k_anomalous_dimension, estimated_dimension_contribution):
#             #     if labels.sum() != 0.0 and not np.isnan(contribution_per_var).all():
#             #         interpretability = labels[top_k_index].sum() / labels.sum()
#             #         interpretability_list.append(interpretability)
#             #     else:
#             #         interpretability_list.append(np.nan)
#             # # interpretability_scores = np.sqrt(np.power(multivariate_labels-anomaly_scores_per_var,2).sum(axis=1))
#             # interpretability_scores = np.array(interpretability_list)
#             #
#             # return interpretability_scores[y_true == 1].mean()
#         return results_dict
#
#     @property
#     def name(self) -> str:
#         return f'Interpretability_Unconditional_NDCG_Hit_{self.max_k}_Score'.upper()
#
#     def get_name_template(self) -> str:
#         return 'Interpretability_Unconditional_NDCG_Hit_{k}_Score'.upper()
# class InterpretabilityUnconditionalHitKScore:
#     """Takes an anomaly scoring and ground truth labels to compute and apply a threshold to the scoring.
#
#     Subclasses of this abstract base class define different strategies to put a threshold over the anomaly scorings.
#     All strategies produce binary labels (0 or 1; 1 for anomalous) in the form of an integer NumPy array.
#     The strategy :class:`~timeeval.metrics.thresholding.NoThresholding` is a special no-op strategy that checks for
#     already existing binary labels and keeps them untouched. This allows applying the metrics on existing binary
#     classification results.
#     """
#
#     def __init__(self, max_k, buffer=10) -> None:
#         self.max_k: Optional[int] = max_k
#         self.buffer = buffer
#
#     def score_for_different_k(self, y_true_multivariate: np.ndarray, dimension_contribution: np.ndarray) -> Dict:
#         assert y_true_multivariate.ndim == 2
#         assert dimension_contribution.ndim == 2
#
#         results_dict = dict() # Save results for all k
#
#         y_true = (y_true_multivariate.sum(axis=1) >= 1).astype(float)
#         # y_score_per_var[np.isnan(y_score_per_var)] = 0.0
#         # y_score_per_var = convert_raw_anomaly_score_per_var_to_contribution_percentage(y_score_per_var)
#         estimated_dimension_contribution = estimate_dimension_contribution_with_a_buffer(dimension_contribution, buffer=self.buffer)
#         dimension_contribution_ranking = np.argsort(estimated_dimension_contribution, axis=1)
#
#         for k in range(1, self.max_k + 1):
#             interpretability_score = calculate_unconditional_interpretability_scores(y_true_multivariate, dimension_contribution_ranking, top_k=k)
#             results_dict[k] = interpretability_score
#             # for labels, top_k_index, contribution_per_var in zip(y_true_multivariate, top_k_anomalous_dimension, estimated_dimension_contribution):
#             #     if labels.sum() != 0.0 and not np.isnan(contribution_per_var).all():
#             #         interpretability = labels[top_k_index].sum() / labels.sum()
#             #         interpretability_list.append(interpretability)
#             #     else:
#             #         interpretability_list.append(np.nan)
#             # # interpretability_scores = np.sqrt(np.power(multivariate_labels-anomaly_scores_per_var,2).sum(axis=1))
#             # interpretability_scores = np.array(interpretability_list)
#             #
#             # return interpretability_scores[y_true == 1].mean()
#         return results_dict
#
#     @property
#     def name(self) -> str:
#         return f'Interpretability_Unconditional_Hit_Max_{self.max_k}_Score'.upper()
#
#     def get_name_template(self) -> str:
#         return 'Interpretability_Unconditional_Hit_{k}_Score'.upper()
#
# class InterpretabilityUnconditionalAllKScore:
#     """Takes an anomaly scoring and ground truth labels to compute and apply a threshold to the scoring.
#
#     Subclasses of this abstract base class define different strategies to put a threshold over the anomaly scorings.
#     All strategies produce binary labels (0 or 1; 1 for anomalous) in the form of an integer NumPy array.
#     The strategy :class:`~timeeval.metrics.thresholding.NoThresholding` is a special no-op strategy that checks for
#     already existing binary labels and keeps them untouched. This allows applying the metrics on existing binary
#     classification results.
#     """
#
#     def __init__(self, max_k, buffer=10) -> None:
#         self.max_k: Optional[int] = max_k
#         self.buffer = buffer
#
#     @property
#     def name(self) -> str:
#         return f'Interpretability_Unconditional_All_Hit_{self.max_k}_Score'.upper()
#
#     def get_name_template(self) -> str:
#         return 'Interpretability_Unconditional_All_Hit_{k}_Score'.upper()
#
#     def score_for_different_k(self, y_true_multivariate: np.ndarray, dimension_contribution: np.ndarray) -> Dict:
#         assert y_true_multivariate.ndim == 2
#         assert dimension_contribution.ndim == 2
#
#         results_dict = dict() # Save results for all k
#
#         y_true = (y_true_multivariate.sum(axis=1) >= 1).astype(float)
#         # y_score_per_var[np.isnan(y_score_per_var)] = 0.0
#         # y_score_per_var = convert_raw_anomaly_score_per_var_to_contribution_percentage(y_score_per_var)
#         estimated_dimension_contribution = estimate_dimension_contribution_with_a_buffer(dimension_contribution, buffer=self.buffer)
#         dimension_contribution_ranking = np.argsort(estimated_dimension_contribution, axis=1)
#
#         for k in range(1, self.max_k + 1):
#             interpretability_score = calculate_unconditional_interpretability_scores(y_true_multivariate, dimension_contribution_ranking, top_k=k)
#             results_dict[k] = interpretability_score
#             # for labels, top_k_index, contribution_per_var in zip(y_true_multivariate, top_k_anomalous_dimension, estimated_dimension_contribution):
#             #     if labels.sum() != 0.0 and not np.isnan(contribution_per_var).all():
#             #         interpretability = labels[top_k_index].sum() / labels.sum()
#             #         interpretability_list.append(interpretability)
#             #     else:
#             #         interpretability_list.append(np.nan)
#             # # interpretability_scores = np.sqrt(np.power(multivariate_labels-anomaly_scores_per_var,2).sum(axis=1))
#             # interpretability_scores = np.array(interpretability_list)
#             #
#             # return interpretability_scores[y_true == 1].mean()
#         if np.version.version >= '2.0.0':
#             all_k_combine_score = abs(np.trapezoid(list(results_dict.values()), list(results_dict.keys())))/self.max_k
#         else:
#             all_k_combine_score = abs(np.trapz(list(results_dict.values()), list(results_dict.keys()))) / self.max_k
#         results_dict = dict({self.max_k: all_k_combine_score})
#         return results_dict
#
# class InterpretabilityConditionalNDCGHitKScore:
#     """Takes an anomaly scoring and ground truth labels to compute and apply a threshold to the scoring.
#
#     Subclasses of this abstract base class define different strategies to put a threshold over the anomaly scorings.
#     All strategies produce binary labels (0 or 1; 1 for anomalous) in the form of an integer NumPy array.
#     The strategy :class:`~timeeval.metrics.thresholding.NoThresholding` is a special no-op strategy that checks for
#     already existing binary labels and keeps them untouched. This allows applying the metrics on existing binary
#     classification results.
#     """
#
#     def __init__(self, max_k: int, buffer: int=10, n_thresholds=250) -> None:
#         self.max_k: Optional[int] = max_k
#         self.buffer = buffer
#         self.n_thresholds = n_thresholds
#
#
#     # def supports_continuous_scorings(self) -> bool:
#     #     return True
#     @property
#     def name(self) -> str:
#         return f'Interpretability_Conditional_NDCG_Hit_{self.max_k}_Score'.upper()
#
#     def get_name_template(self) -> str:
#         return 'Interpretability_Conditional_NDCG_Hit_{k}_Score'.upper()
#
#     def score_for_different_k(self, y_true_univariate, y_score_univariate, y_true_multivariate: np.ndarray, dimension_contribution_multivariate: np.ndarray) -> Dict:
#         assert y_true_multivariate.ndim == 2
#         assert dimension_contribution_multivariate.ndim == 2
#         assert y_true_univariate.ndim == 1
#         assert y_score_univariate.ndim == 1
#         # fpr, tpr, thresholds = roc_curve(y_true.reshape(-1), y_score.reshape(-1))
#         # result = auc(fpr, tpr)
#
#         y_true = (y_true_multivariate.sum(axis=1)>=1).astype(float)
#         assert (y_true == y_true_univariate).all()
#
#         y_score_univariate_sorted = -np.sort(-y_score_univariate)
#
#         thresholds = []
#         precision = []
#         recall = []
#         pred_for_thresholds = [] # Store predictions for each threshold
#
#         for k, i in enumerate(np.linspace(0, len(y_score_univariate) - 1, self.n_thresholds).astype(int)):
#             threshold = y_score_univariate_sorted[i]
#             pred = (y_score_univariate >= threshold).astype(float)
#             tp = np.sum((pred == 1) & (y_true_univariate == 1))
#             fp = np.sum((pred == 1) & (y_true_univariate == 0))
#             fn = np.sum((pred == 0) & (y_true_univariate == 1))
#             precision.append(tp / (tp + fp) if (tp + fp) > 0 else 0)
#             recall.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
#             thresholds.append(threshold)
#             pred_for_thresholds.append(pred)
#         thresholds = np.array(thresholds)
#         precision = np.array(precision)
#         recall = np.array(recall)
#         pred_for_thresholds = np.stack(pred_for_thresholds, axis=1)
#
#
#         estimated_dimension_contribution = estimate_dimension_contribution_with_a_buffer(dimension_contribution_multivariate,
#                                                                                          buffer=self.buffer)
#         estimated_dimension_contribution = estimated_dimension_contribution/estimated_dimension_contribution.sum(axis=1, keepdims=True)
#         dimension_contribution_ranking = np.argsort(estimated_dimension_contribution, axis=1)
#
#
#
#         # precision, recall, thresholds = precision_recall_curve(y_true_univariate, np.round(y_score_univariate,3))
#         # # print('precision:', precision.shape, ' recall:', recall.shape, ' thresholds:', thresholds.shape)
#         # metric_matrix = np.stack((precision[:-1], recall[:-1], thresholds), axis=1)
#         # precision_recal_sum = metric_matrix[:, 0] + metric_matrix[:, 1]
#         # metric_matrix = metric_matrix[precision_recal_sum != 0]
#         # precision = metric_matrix[:, 0]
#         # recall = metric_matrix[:, 1]
#         # thresholds = metric_matrix[:, 2]
#
#         # f1_scores = 2 * (precision * recall) / (precision + recall)
#         # max_f1_score = np.nanmax(f1_scores)
#         # optimal_threshold = thresholds[np.nanargmax(f1_scores)]
#
#         # results_dict = dict({k: np.zeros(len(thresholds)) for k in range(1, self.max_k+1)})
#         results_dict = dict({self.max_k: np.zeros(len(thresholds))}) # Save results for all k
#
#         # for threshold_index, threshold in enumerate(thresholds):
#         #     y_pred = np.array(y_score_univariate >= threshold, dtype=float)
#         start_time = time.time()
#
#         for k in range(self.max_k, self.max_k + 1):
#             ndcg_score_list = _ndcg_sample_scores(y_true_multivariate, estimated_dimension_contribution, k=self.max_k)
#             pred_overlapped_for_thresholds = np.apply_along_axis(lambda pred: (pred == 1.0) & (y_true_univariate == 1), axis=0, arr=pred_for_thresholds)
#             score_for_thresholds = np.where(pred_overlapped_for_thresholds == 1, ndcg_score_list.reshape(-1,1), np.nan)
#
#             all_nan_threshold_indexes = np.where(np.all(np.isnan(score_for_thresholds), axis=0))[0]
#             score_for_thresholds[:, all_nan_threshold_indexes] = 0.0
#             score_for_thresholds = np.nanmean(score_for_thresholds, axis=0)
#             results_dict[k] = score_for_thresholds
#             # score_for_thresholds = np.apply_along_axis(lambda pred: ndcg_score(y_true_multivariate[pred == 1.0], estimated_dimension_contribution[pred == 1.0], k=k) if (pred == 1.0).sum() > 0 else 0.0, axis=0, arr=pred_for_thresholds)
#             # for threshold_index, threshold in enumerate(thresholds):
#             #     detected_tp = (pred_for_thresholds[:,threshold_index] == 1.0) & (y_true_univariate == 1.0)
#             #     if detected_tp.sum() == 0:
#             #         score = 0.0
#             #     else:
#             #         score = ndcg_score(y_true_multivariate[detected_tp], estimated_dimension_contribution[detected_tp], k=k)
#             #     results_dict[k][threshold_index] = score
#         end_time = time.time()
#         # print(f"Time taken to calculate interpretability scores for all thresholds and k values: {end_time - start_time:.2f} seconds")
#
#         for k, interpretability_scores in results_dict.items():
#             new_precision = precision * interpretability_scores
#             new_recall = recall * interpretability_scores
#             new_recall_sorted_indices = np.argsort(new_recall)
#             new_recall = new_recall[new_recall_sorted_indices]
#             new_precision = new_precision[new_recall_sorted_indices]
#             results_dict[k] = abs(np.trapz(new_precision, new_recall))
#         # results_dict = {k: abs(np.trapz(interpretability_scores*precision, interpretability_scores*recall)) }
#         return results_dict
#
# class AVUSI:
#     """Takes an anomaly scoring and ground truth labels to compute and apply a threshold to the scoring.
#
#     Subclasses of this abstract base class define different strategies to put a threshold over the anomaly scorings.
#     All strategies produce binary labels (0 or 1; 1 for anomalous) in the form of an integer NumPy array.
#     The strategy :class:`~timeeval.metrics.thresholding.NoThresholding` is a special no-op strategy that checks for
#     already existing binary labels and keeps them untouched. This allows applying the metrics on existing binary
#     classification results.
#     """
#
#     def __init__(self, max_k: int, slope:int=64, buffer: int = 10, n_thresholds=250) -> None:
#         self.max_k: Optional[int] = max_k
#         self.buffer = buffer
#         self.n_thresholds = n_thresholds
#         self.slope = slope
#
#         # def supports_continuous_scorings(self) -> bool:
#         #     return True
#
#     @property
#     def name(self) -> str:
#         return f'AVUSI_{self.max_k}_Score'.upper()
#
#     def get_name_template(self) -> str:
#         return 'AVUSI_{k}_Score'.upper()
#
#     def score_for_different_k(self, y_true_univariate, y_score_univariate, y_true_multivariate: np.ndarray, dimension_contribution_multivariate: np.ndarray) -> Dict:
#         assert y_true_multivariate.ndim == 2
#         assert dimension_contribution_multivariate.ndim == 2
#         assert y_true_univariate.ndim == 1
#         assert y_score_univariate.ndim == 1
#         # fpr, tpr, thresholds = roc_curve(y_true.reshape(-1), y_score.reshape(-1))
#         # result = auc(fpr, tpr)
#
#         y_true = (y_true_multivariate.sum(axis=1) >= 1).astype(float)
#         assert (y_true == y_true_univariate).all()
#
#         # y_score_univariate_sorted = -np.sort(-y_score_univariate)
#         #
#         # thresholds = []
#         # precision = []
#         # recall = []
#         # pred_for_thresholds = []  # Store predictions for each threshold
#         #
#         # for k, i in enumerate(np.linspace(0, len(y_score_univariate) - 1, self.n_thresholds).astype(int)):
#         #     threshold = y_score_univariate_sorted[i]
#         #     pred = (y_score_univariate >= threshold).astype(float)
#         #     tp = np.sum((pred == 1) & (y_true_univariate == 1))
#         #     fp = np.sum((pred == 1) & (y_true_univariate == 0))
#         #     fn = np.sum((pred == 0) & (y_true_univariate == 1))
#         #     precision.append(tp / (tp + fp) if (tp + fp) > 0 else 0)
#         #     recall.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
#         #     thresholds.append(threshold)
#         #     pred_for_thresholds.append(pred)
#         # thresholds = np.array(thresholds)
#         # precision = np.array(precision)
#         # recall = np.array(recall)
#         # pred_for_thresholds = np.stack(pred_for_thresholds, axis=1)
#
#         estimated_dimension_contribution = estimate_dimension_contribution_with_a_buffer(
#             dimension_contribution_multivariate,
#             buffer=self.buffer)
#         estimated_dimension_contribution = estimated_dimension_contribution / estimated_dimension_contribution.sum(
#             axis=1, keepdims=True)
#         # dimension_contribution_ranking = np.argsort(estimated_dimension_contribution, axis=1)
#
#         interpretability_scores = _ndcg_sample_scores(y_true_multivariate, estimated_dimension_contribution, k=self.max_k)
#         interpretability_scores[y_true_univariate == 0] = 1.0  # Set interpretability scores of normal samples to the minimum score
#         new_anomaly_scores = y_score_univariate * interpretability_scores
#
#         device = 'cuda' if torch.cuda.is_available() else 'cpu'
#         vus = VUSTorch(slope_size=self.slope, device=device)
#         value_data, timing = vus.compute(torch.from_numpy(y_true).to(device), torch.from_numpy(np.round(new_anomaly_scores, decimals=2)).to(device))
#         result_dict = {self.max_k: value_data[0].item()}
#         return result_dict
#
#     # def supports_continuous_scorings(self) -> bool:
#     #     return True
#
# class FFVUS_With_Interpretability_Combination:
#     """Takes an anomaly scoring and ground truth labels to compute and apply a threshold to the scoring.
#
#     Subclasses of this abstract base class define different strategies to put a threshold over the anomaly scorings.
#     All strategies produce binary labels (0 or 1; 1 for anomalous) in the form of an integer NumPy array.
#     The strategy :class:`~timeeval.metrics.thresholding.NoThresholding` is a special no-op strategy that checks for
#     already existing binary labels and keeps them untouched. This allows applying the metrics on existing binary
#     classification results.
#     """
#
#     def __init__(self, max_k: int, slope:int=64, buffer: int = 10, n_thresholds=250) -> None:
#         self.max_k: Optional[int] = max_k
#         self.buffer = buffer
#         self.n_thresholds = n_thresholds
#         self.slope = slope
#
#         # def supports_continuous_scorings(self) -> bool:
#         #     return True
#
#     @property
#     def name(self) -> str:
#         return f'Interpretability_Conditional_Volumn_PR_with_NDCG_Hit_{self.max_k}_Score_Combination'.upper()
#
#     def get_name_template(self) -> str:
#         return 'Interpretability_Conditional_Volumn_PR_with_NDCG_Hit_{k}_Score_Combination'.upper()
#
#     def score_for_different_k(self, y_true_univariate, y_score_univariate, y_true_multivariate: np.ndarray, dimension_contribution_multivariate: np.ndarray) -> Dict:
#         assert y_true_multivariate.ndim == 2
#         assert dimension_contribution_multivariate.ndim == 2
#         assert y_true_univariate.ndim == 1
#         assert y_score_univariate.ndim == 1
#         # fpr, tpr, thresholds = roc_curve(y_true.reshape(-1), y_score.reshape(-1))
#         # result = auc(fpr, tpr)
#
#         y_true = (y_true_multivariate.sum(axis=1) >= 1).astype(float)
#         assert (y_true == y_true_univariate).all()
#
#         # y_score_univariate_sorted = -np.sort(-y_score_univariate)
#         #
#         # thresholds = []
#         # precision = []
#         # recall = []
#         # pred_for_thresholds = []  # Store predictions for each threshold
#         #
#         # for k, i in enumerate(np.linspace(0, len(y_score_univariate) - 1, self.n_thresholds).astype(int)):
#         #     threshold = y_score_univariate_sorted[i]
#         #     pred = (y_score_univariate >= threshold).astype(float)
#         #     tp = np.sum((pred == 1) & (y_true_univariate == 1))
#         #     fp = np.sum((pred == 1) & (y_true_univariate == 0))
#         #     fn = np.sum((pred == 0) & (y_true_univariate == 1))
#         #     precision.append(tp / (tp + fp) if (tp + fp) > 0 else 0)
#         #     recall.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
#         #     thresholds.append(threshold)
#         #     pred_for_thresholds.append(pred)
#         # thresholds = np.array(thresholds)
#         # precision = np.array(precision)
#         # recall = np.array(recall)
#         # pred_for_thresholds = np.stack(pred_for_thresholds, axis=1)
#
#         estimated_dimension_contribution = estimate_dimension_contribution_with_a_buffer(
#             dimension_contribution_multivariate,
#             buffer=self.buffer)
#         estimated_dimension_contribution = estimated_dimension_contribution / estimated_dimension_contribution.sum(
#             axis=1, keepdims=True)
#         # dimension_contribution_ranking = np.argsort(estimated_dimension_contribution, axis=1)
#
#
#         interpretability_scores = _ndcg_sample_scores(y_true_multivariate, estimated_dimension_contribution, k=self.max_k)
#
#         L_list = np.linspace(0, 1, 50)
#         vus_pr_list = []
#         for L in L_list:
#             interpretability_scores[y_true_univariate == 0] = L  # Set interpretability scores of normal samples to the minimum score
#             new_anomaly_scores = y_score_univariate * interpretability_scores
#
#             device = 'cuda' if torch.cuda.is_available() else 'cpu'
#             vus = VUSTorch(slope_size=self.slope, device=device)
#             value_data, timing = vus.compute(torch.from_numpy(y_true).to(device), torch.from_numpy(np.round(new_anomaly_scores, decimals=2)).to(device))
#             vus_pr_list.append(value_data[0].item())
#         result_dict = {self.max_k: np.mean(vus_pr_list)}
#         result_dict['L_list'] = L_list
#         result_dict['vus_pr_list'] = vus_pr_list
#         return result_dict
#     # def supports_continuous_scorings(self) -> bool:
#     #     return True
#
# class InterpretabilityConditionalHitKScore:
#     """Takes an anomaly scoring and ground truth labels to compute and apply a threshold to the scoring.
#
#     Subclasses of this abstract base class define different strategies to put a threshold over the anomaly scorings.
#     All strategies produce binary labels (0 or 1; 1 for anomalous) in the form of an integer NumPy array.
#     The strategy :class:`~timeeval.metrics.thresholding.NoThresholding` is a special no-op strategy that checks for
#     already existing binary labels and keeps them untouched. This allows applying the metrics on existing binary
#     classification results.
#     """
#
#     def __init__(self, max_k: int, buffer: int=10, n_thresholds=250) -> None:
#         self.max_k: Optional[int] = max_k
#         self.buffer = buffer
#         self.n_thresholds = n_thresholds
#
#
#     # def supports_continuous_scorings(self) -> bool:
#     #     return True
#     @property
#     def name(self) -> str:
#         return f'Interpretability_Conditional_Hit_{self.max_k}_Score'.upper()
#
#     def get_name_template(self) -> str:
#         return 'Interpretability_Conditional_Hit_{k}_Score'.upper()
#
#     def score_for_different_k(self, y_true_univariate, y_score_univariate, y_true_multivariate: np.ndarray, dimension_contribution_multivariate: np.ndarray) -> Dict:
#         assert y_true_multivariate.ndim == 2
#         assert dimension_contribution_multivariate.ndim == 2
#         assert y_true_univariate.ndim == 1
#         assert y_score_univariate.ndim == 1
#         # fpr, tpr, thresholds = roc_curve(y_true.reshape(-1), y_score.reshape(-1))
#         # result = auc(fpr, tpr)
#
#         y_true = (y_true_multivariate.sum(axis=1)>=1).astype(float)
#         assert (y_true == y_true_univariate).all()
#
#         y_score_univariate_sorted = -np.sort(-y_score_univariate)
#
#         thresholds = []
#         precision = []
#         recall = []
#         pred_for_thresholds = [] # Store predictions for each threshold
#
#         for k, i in enumerate(np.linspace(0, len(y_score_univariate) - 1, self.n_thresholds).astype(int)):
#             threshold = y_score_univariate_sorted[i]
#             pred = (y_score_univariate >= threshold).astype(float)
#             tp = np.sum((pred == 1) & (y_true_univariate == 1))
#             fp = np.sum((pred == 1) & (y_true_univariate == 0))
#             fn = np.sum((pred == 0) & (y_true_univariate == 1))
#             precision.append(tp / (tp + fp) if (tp + fp) > 0 else 0)
#             recall.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
#             thresholds.append(threshold)
#             pred_for_thresholds.append(pred)
#         thresholds = np.array(thresholds)
#         precision = np.array(precision)
#         recall = np.array(recall)
#         pred_for_thresholds = np.stack(pred_for_thresholds, axis=1)
#
#
#         estimated_dimension_contribution = estimate_dimension_contribution_with_a_buffer(dimension_contribution_multivariate,
#                                                                                          buffer=self.buffer)
#         dimension_contribution_ranking = np.argsort(estimated_dimension_contribution, axis=1)
#
#
#
#         # precision, recall, thresholds = precision_recall_curve(y_true_univariate, np.round(y_score_univariate,3))
#         # # print('precision:', precision.shape, ' recall:', recall.shape, ' thresholds:', thresholds.shape)
#         # metric_matrix = np.stack((precision[:-1], recall[:-1], thresholds), axis=1)
#         # precision_recal_sum = metric_matrix[:, 0] + metric_matrix[:, 1]
#         # metric_matrix = metric_matrix[precision_recal_sum != 0]
#         # precision = metric_matrix[:, 0]
#         # recall = metric_matrix[:, 1]
#         # thresholds = metric_matrix[:, 2]
#
#         # f1_scores = 2 * (precision * recall) / (precision + recall)
#         # max_f1_score = np.nanmax(f1_scores)
#         # optimal_threshold = thresholds[np.nanargmax(f1_scores)]
#
#         # results_dict = dict({k: np.zeros(len(thresholds)) for k in range(1, self.max_k+1)})
#
#         # for threshold_index, threshold in enumerate(thresholds):
#         #     y_pred = np.array(y_score_univariate >= threshold, dtype=float)
#         # start_time = time.time()
#         # for k in range(1, self.max_k + 1):
#         results_dict = calculate_interpretability_scores(pred_for_thresholds, y_true_multivariate, dimension_contribution_ranking, self.max_k)
#             # results_dict[k] = score_for_thresholds
#         # end_time = time.time()
#         # print(f"Time taken to calculate interpretability scores for all thresholds and k values: {end_time - start_time:.2f} seconds")
#
#         for k, interpretability_scores in results_dict.items():
#             new_precision = precision * interpretability_scores
#             new_recall = recall * interpretability_scores
#             new_recall_sorted_indices = np.argsort(new_recall)
#             new_recall = new_recall[new_recall_sorted_indices]
#             new_precision = new_precision[new_recall_sorted_indices]
#             results_dict[k] = abs(np.trapz(new_precision, new_recall))
#         # results_dict = {k: abs(np.trapz(interpretability_scores*precision, interpretability_scores*recall)) }
#         return results_dict
#
# class InterpretabilityConditionalAllKScore:
#     """Takes an anomaly scoring and ground truth labels to compute and apply a threshold to the scoring.
#
#     Subclasses of this abstract base class define different strategies to put a threshold over the anomaly scorings.
#     All strategies produce binary labels (0 or 1; 1 for anomalous) in the form of an integer NumPy array.
#     The strategy :class:`~timeeval.metrics.thresholding.NoThresholding` is a special no-op strategy that checks for
#     already existing binary labels and keeps them untouched. This allows applying the metrics on existing binary
#     classification results.
#     """
#
#     def __init__(self, max_k: int, buffer: int=10, n_thresholds=250) -> None:
#         self.max_k: Optional[int] = max_k
#         self.buffer = buffer
#         self.n_thresholds = n_thresholds
#
#
#     # def supports_continuous_scorings(self) -> bool:
#     #     return True
#     @property
#     def name(self) -> str:
#         return f'Interpretability_Conditional_All_Hit_{self.max_k}_Score'.upper()
#
#     def get_name_template(self) -> str:
#         return 'Interpretability_Conditional_All_Hit_{k}_Score'.upper()
#
#     def score_for_different_k(self, y_true_univariate, y_score_univariate, y_true_multivariate: np.ndarray, dimension_contribution_multivariate: np.ndarray) -> Dict:
#         assert y_true_multivariate.ndim == 2
#         assert dimension_contribution_multivariate.ndim == 2
#         assert y_true_univariate.ndim == 1
#         assert y_score_univariate.ndim == 1
#         # fpr, tpr, thresholds = roc_curve(y_true.reshape(-1), y_score.reshape(-1))
#         # result = auc(fpr, tpr)
#
#         y_true = (y_true_multivariate.sum(axis=1)>=1).astype(float)
#         assert (y_true == y_true_univariate).all()
#
#         y_score_univariate_sorted = -np.sort(-y_score_univariate)
#
#         thresholds = []
#         precision = []
#         recall = []
#         pred_for_thresholds = [] # Store predictions for each threshold
#
#         for k, i in enumerate(np.linspace(0, len(y_score_univariate) - 1, self.n_thresholds).astype(int)):
#             threshold = y_score_univariate_sorted[i]
#             pred = (y_score_univariate >= threshold).astype(float)
#             tp = np.sum((pred == 1) & (y_true_univariate == 1))
#             fp = np.sum((pred == 1) & (y_true_univariate == 0))
#             fn = np.sum((pred == 0) & (y_true_univariate == 1))
#             precision.append(tp / (tp + fp) if (tp + fp) > 0 else 0)
#             recall.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
#             thresholds.append(threshold)
#             pred_for_thresholds.append(pred)
#         thresholds = np.array(thresholds)
#         precision = np.array(precision)
#         recall = np.array(recall)
#         pred_for_thresholds = np.stack(pred_for_thresholds, axis=1)
#
#
#         estimated_dimension_contribution = estimate_dimension_contribution_with_a_buffer(dimension_contribution_multivariate,
#                                                                                          buffer=self.buffer)
#         dimension_contribution_ranking = np.argsort(estimated_dimension_contribution, axis=1)
#
#
#
#         # precision, recall, thresholds = precision_recall_curve(y_true_univariate, np.round(y_score_univariate,3))
#         # # print('precision:', precision.shape, ' recall:', recall.shape, ' thresholds:', thresholds.shape)
#         # metric_matrix = np.stack((precision[:-1], recall[:-1], thresholds), axis=1)
#         # precision_recal_sum = metric_matrix[:, 0] + metric_matrix[:, 1]
#         # metric_matrix = metric_matrix[precision_recal_sum != 0]
#         # precision = metric_matrix[:, 0]
#         # recall = metric_matrix[:, 1]
#         # thresholds = metric_matrix[:, 2]
#
#         # f1_scores = 2 * (precision * recall) / (precision + recall)
#         # max_f1_score = np.nanmax(f1_scores)
#         # optimal_threshold = thresholds[np.nanargmax(f1_scores)]
#
#         # results_dict = dict({k: np.zeros(len(thresholds)) for k in range(1, self.max_k+1)})
#
#         # for threshold_index, threshold in enumerate(thresholds):
#         #     y_pred = np.array(y_score_univariate >= threshold, dtype=float)
#         # start_time = time.time()
#         # for k in range(1, self.max_k + 1):
#         results_dict = calculate_interpretability_scores(pred_for_thresholds, y_true_multivariate, dimension_contribution_ranking, self.max_k)
#             # results_dict[k] = score_for_thresholds
#         # end_time = time.time()
#         # print(f"Time taken to calculate interpretability scores for all thresholds and k values: {end_time - start_time:.2f} seconds")
#
#         for k, interpretability_scores in results_dict.items():
#             new_precision = precision * interpretability_scores
#             new_recall = recall * interpretability_scores
#             new_recall_sorted_indices = np.argsort(new_recall)
#             new_recall = new_recall[new_recall_sorted_indices]
#             new_precision = new_precision[new_recall_sorted_indices]
#             results_dict[k] = abs(np.trapz(new_precision, new_recall))
#         # results_dict = {k: abs(np.trapz(interpretability_scores, recall)) for k, interpretability_scores in results_dict.items()}
#         combine_all_k = abs(np.trapz(list(results_dict.values()), list(results_dict.keys())))/self.max_k
#         results_dict = dict({self.max_k: combine_all_k})
#         # results_dict = {k: calculate_max_f1_scores_with_interpretability(precision*interpretability_scores, recall*interpretability_scores) for k, interpretability_scores in results_dict.items()}
#         return results_dict
#
#
# # class InterpretabilityConditionalHitKScoreUpdate:
# #     """Takes an anomaly scoring and ground truth labels to compute and apply a threshold to the scoring.
# #
# #     Subclasses of this abstract base class define different strategies to put a threshold over the anomaly scorings.
# #     All strategies produce binary labels (0 or 1; 1 for anomalous) in the form of an integer NumPy array.
# #     The strategy :class:`~timeeval.metrics.thresholding.NoThresholding` is a special no-op strategy that checks for
# #     already existing binary labels and keeps them untouched. This allows applying the metrics on existing binary
# #     classification results.
# #     """
# #
# #     def __init__(self, top_k) -> None:
# #         self.top_k: Optional[int] = top_k
# #
# #     def score_and_output_details(self, y_true_univariate, y_score_univariate, y_true_multivariate: np.ndarray, y_score_per_var: np.ndarray) -> Tuple[float, float, float]:
# #         assert y_true_multivariate.ndim == 2
# #         assert y_score_per_var.ndim == 2
# #         assert y_true_univariate.ndim == 1
# #         assert y_score_univariate.ndim == 1
# #         # fpr, tpr, thresholds = roc_curve(y_true.reshape(-1), y_score.reshape(-1))
# #         # result = auc(fpr, tpr)
# #
# #         y_true = (y_true_multivariate.sum(axis=1)>=1).astype(float)
# #         assert (y_true == y_true_univariate).all()
# #
# #         precision, recall, thresholds = precision_recall_curve(y_true_univariate, np.round(y_score_univariate,2))
# #         # f1_scores = 2 * (precision * recall) / (precision + recall)
# #         # max_f1_score = np.max(f1_scores)
# #         # optimal_threshold = thresholds[np.argmax(f1_scores)]
# #
# #         # if np.isnan(max_f1_score):
# #         #     max_f1_score = 0.0
# #         #     optimal_threshold = np.nan
# #
# #         # detected_anomalies = np.array(y_score_univariate >= optimal_threshold, dtype=float)
# #         #
# #         # anomaly_scores_per_var_ranking = np.argsort(y_score_per_var, axis=1)
# #         # top_k_anomalous_dimension = anomaly_scores_per_var_ranking[:, -self.top_k:]
# #         # interpretability_list = []
# #         # for labels, top_k_index, detected_anomaly in zip(y_true_multivariate, top_k_anomalous_dimension, detected_anomalies):
# #         #     if labels.sum() != 0.0 and detected_anomaly != 0.0:
# #         #         interpretability = labels[top_k_index].sum() / labels.sum()
# #         #         interpretability_list.append(interpretability)
# #         #     else:
# #         #         interpretability_list.append(0)
# #         # # interpretability_scores = np.sqrt(np.power(multivariate_labels-anomaly_scores_per_var,2).sum(axis=1))
# #         # interpretability_scores = np.array(interpretability_list)
# #         #
# #         # return interpretability_scores[y_true == 1.0].mean(), max_f1_score, optimal_threshold
# #
# #         all_results = []
# #         for optimal_threshold in thresholds:
# #             y_pred = np.array(y_score_univariate >= optimal_threshold, dtype=float)
# #             result = calculate_interpretability_scores(y_pred, y_true_multivariate, y_score_per_var, self.top_k)
# #             all_results.append((result, 0, optimal_threshold))
# #             # detected_anomalies = np.array(y_score_univariate >= optimal_threshold, dtype=float)
# #             #
# #             # anomaly_scores_per_var_ranking = np.argsort(y_score_per_var, axis=1)
# #             # top_k_anomalous_dimension = anomaly_scores_per_var_ranking[:, -self.top_k:]
# #             # interpretability_list = []
# #             # for labels, top_k_index, detected_anomaly in zip(y_true_multivariate, top_k_anomalous_dimension,
# #             #                                                  detected_anomalies):
# #             #     if labels.sum() != 0.0 and detected_anomaly != 0.0:
# #             #         interpretability = labels[top_k_index].sum() / labels.sum()
# #             #         interpretability_list.append(interpretability)
# #             #     else:
# #             #         interpretability_list.append(0)
# #             # # interpretability_scores = np.sqrt(np.power(multivariate_labels-anomaly_scores_per_var,2).sum(axis=1))
# #             # interpretability_scores = np.array(interpretability_list)
# #             # all_results.append((interpretability_scores[y_true == 1.0].mean(), max_f1_score, optimal_threshold))
# #
# #         all_results = np.array(all_results)
# #         return np.trapz(all_results[:, 0], all_results[:, 2]), 0, 0
# #
# #     # def supports_continuous_scorings(self) -> bool:
# #     #     return True
# #     @property
# #     def name(self) -> str:
# #         return f'Interpretability_Conditional_Hit_{self.top_k}_Score_Update'.upper()
#
#
# class InterpretabilityVUSConditionalHitKScore:
#     """Takes an anomaly scoring and ground truth labels to compute and apply a threshold to the scoring.
#
#     Subclasses of this abstract base class define different strategies to put a threshold over the anomaly scorings.
#     All strategies produce binary labels (0 or 1; 1 for anomalous) in the form of an integer NumPy array.
#     The strategy :class:`~timeeval.metrics.thresholding.NoThresholding` is a special no-op strategy that checks for
#     already existing binary labels and keeps them untouched. This allows applying the metrics on existing binary
#     classification results.
#     """
#
#     def __init__(self, max_k: int, buffer: int=10, n_thresholds=250, slope=64) -> None:
#         self.max_k: Optional[int] = max_k
#         self.buffer = buffer
#         self.n_thresholds = n_thresholds
#         self.slope = slope
#
#
#     # def supports_continuous_scorings(self) -> bool:
#     #     return True
#     @property
#     def name(self) -> str:
#         return f'Interpretability_Volumn_Conditional_Hit_Max_{self.max_k}_Score'.upper()
#
#     def get_name_template(self) -> str:
#         return 'Interpretability_Volumn_Conditional_Hit_{k}_Score'.upper()
#
#     def score_for_different_k(self, y_true_univariate, y_score_univariate, y_true_multivariate: np.ndarray, dimension_contribution_multivariate: np.ndarray) -> Dict:
#         assert y_true_multivariate.ndim == 2
#         assert dimension_contribution_multivariate.ndim == 2
#         assert y_true_univariate.ndim == 1
#         assert y_score_univariate.ndim == 1
#         # fpr, tpr, thresholds = roc_curve(y_true.reshape(-1), y_score.reshape(-1))
#         # result = auc(fpr, tpr)
#
#         y_true = (y_true_multivariate.sum(axis=1)>=1).astype(float)
#         assert (y_true == y_true_univariate).all()
#
#         anomalies_indices = range_convers_new(y_true)
#
#         y_true_univariate_extended = sequencing(y_true, anomalies_indices, window=self.slope)
#
#         y_score_univariate_sorted = -np.sort(-y_score_univariate)
#
#
#
#         thresholds = []
#         precision = []
#         recall = []
#         pred_for_thresholds = [] # Store predictions for each threshold
#
#         for k, i in enumerate(np.linspace(0, len(y_score_univariate) - 1, self.n_thresholds).astype(int)):
#             threshold = y_score_univariate_sorted[i]
#             pred = (y_score_univariate >= threshold).astype(float)
#             tp = np.sum((pred == 1) & (y_true_univariate == 1))
#             fp = np.sum((pred == 1) & (y_true_univariate == 0))
#             fn = np.sum((pred == 0) & (y_true_univariate == 1))
#             precision.append(tp / (tp + fp) if (tp + fp) > 0 else 0)
#             recall.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
#             thresholds.append(threshold)
#             pred_for_thresholds.append(pred)
#         thresholds = np.array(thresholds)
#         precision = np.array(precision)
#         recall = np.array(recall)
#         pred_for_thresholds = np.stack(pred_for_thresholds, axis=1)
#
#
#         estimated_dimension_contribution = estimate_dimension_contribution_with_a_buffer(dimension_contribution_multivariate,
#                                                                                          buffer=self.buffer)
#         dimension_contribution_ranking = np.argsort(estimated_dimension_contribution, axis=1)
#
#
#
#         # precision, recall, thresholds = precision_recall_curve(y_true_univariate, np.round(y_score_univariate,3))
#         # # print('precision:', precision.shape, ' recall:', recall.shape, ' thresholds:', thresholds.shape)
#         # metric_matrix = np.stack((precision[:-1], recall[:-1], thresholds), axis=1)
#         # precision_recal_sum = metric_matrix[:, 0] + metric_matrix[:, 1]
#         # metric_matrix = metric_matrix[precision_recal_sum != 0]
#         # precision = metric_matrix[:, 0]
#         # recall = metric_matrix[:, 1]
#         # thresholds = metric_matrix[:, 2]
#
#         # f1_scores = 2 * (precision * recall) / (precision + recall)
#         # max_f1_score = np.nanmax(f1_scores)
#         # optimal_threshold = thresholds[np.nanargmax(f1_scores)]
#
#         # results_dict = dict({k: np.zeros(len(thresholds)) for k in range(1, self.max_k+1)})
#
#         # for threshold_index, threshold in enumerate(thresholds):
#         #     y_pred = np.array(y_score_univariate >= threshold, dtype=float)
#         # start_time = time.time()
#         # for k in range(1, self.max_k + 1):
#         results_dict = calculate_interpretability_scores(pred_for_thresholds, y_true_multivariate, dimension_contribution_ranking, self.max_k)
#             # results_dict[k] = score_for_thresholds
#         # end_time = time.time()
#         # print(f"Time taken to calculate interpretability scores for all thresholds and k values: {end_time - start_time:.2f} seconds")
#         results_dict = {k: abs(np.trapz(interpretability_scores, recall)) for k, interpretability_scores in results_dict.items()}
#         combine_all_k = abs(np.trapz(list(results_dict.values()), list(results_dict.keys())))
#         results_dict = dict({0: combine_all_k})
#         # results_dict = {k: calculate_max_f1_scores_with_interpretability(precision*interpretability_scores, recall*interpretability_scores) for k, interpretability_scores in results_dict.items()}
#         return results_dict
#
#
#
# def range_convers_new(label):
#     '''
#     input: arrays of binary values
#     output: list of ordered pair [[a0,b0], [a1,b1]... ] of the inputs
#     '''
#     L = []
#     i = 0
#     j = 0
#     while j < len(label):
#         # print(i)
#         while label[i] == 0:
#             i += 1
#             if i >= len(label):  # ?
#                 break  # ?
#         j = i + 1
#         # print('j'+str(j))
#         if j >= len(label):
#             if j == len(label):
#                 L.append((i, j - 1))
#
#             break
#         while label[j] != 0:
#             j += 1
#             if j >= len(label):
#                 L.append((i, j - 1))
#                 break
#         if j >= len(label):
#             break
#         L.append((i, j - 1))
#         i = j
#     return L
#
# def sequencing(x, L, window=5):
#     label = x.copy().astype(float)
#     length = len(label)
#
#     for k in range(len(L)):
#         s = L[k][0]
#         e = L[k][1]
#
#         x1 = np.arange(e + 1, min(e + window // 2 + 1, length))
#         label[x1] += np.sqrt(1 - (x1 - e) / (window))
#
#         x2 = np.arange(max(s - window // 2, 0), s)
#         label[x2] += np.sqrt(1 - (s - x2) / (window))
#
#     label = np.minimum(np.ones(length), label)
#     return label
#
# def new_sequence(label, sequence_original, window):
#     # label is [start,end] of every anomalies
#     # sequence_original is the original labels
#     # window is the tolerance window size
#     a = max(sequence_original[0][0] - window // 2, 0)
#     sequence_new = []
#     for i in range(len(sequence_original) - 1):
#         if sequence_original[i][1] + window // 2 < sequence_original[i + 1][0] - window // 2:
#             sequence_new.append((a, sequence_original[i][1] + window // 2))
#             a = sequence_original[i + 1][0] - window // 2
#     sequence_new.append((a, min(sequence_original[len(sequence_original) - 1][1] + window // 2, len(label) - 1)))
#     return sequence_new
#
# # def calculate_interpretability_scores(y_pred, y_true_multivariate: np.ndarray, contribution_per_var_ranking: np.ndarray, top_k) -> np.ndarray:
# #     top_k_indices_matrix = contribution_per_var_ranking[:, -top_k:]
# #     interpretability_score_df = pd.DataFrame(columns=['score'])
# #     interpretability_score_df['score'] = np.zeros(y_pred.shape[0])
# #     interpretability_score_df['y_pred'] = y_pred
# #     interpretability_score_df['y_true'] = (y_true_multivariate.sum(axis=1) >= 1).astype(float)
# #     interpretability_score_df['detected_anomalous_dimension_count'] = np.take_along_axis(y_true_multivariate, top_k_indices_matrix, axis=1).sum(axis=1)
# #     interpretability_score_df['true_anomalous_dimension_count'] = y_true_multivariate.sum(axis=1)
# #     # interpretability_score_df.loc[interpretability_score_df['y_pred'] == 1.0, 'score'] = interpretability_score_df.loc[interpretability_score_df['y_true'] == 1.0].loc[interpretability_score_df['y_pred'] == 1.0, 'detected_anomalous_dimension_count'] / interpretability_score_df.loc[interpretability_score_df['y_true'] == 1.0].loc[interpretability_score_df['y_pred'] == 1.0, 'true_anomalous_dimension_count']
# #     # interpretability_scores = np.zeros(y_pred.shape[0])
# #     detected_true_anomaly_index = interpretability_score_df[interpretability_score_df['y_pred'] == 1.0].loc[interpretability_score_df['y_true'] == 1.0].index
# #     interpretability_score_df.loc[detected_true_anomaly_index, 'score'] = interpretability_score_df.loc[detected_true_anomaly_index, 'detected_anomalous_dimension_count'].values / interpretability_score_df.loc[detected_true_anomaly_index,'true_anomalous_dimension_count'].values
# #     return interpretability_score_df[(interpretability_score_df['y_pred'] == 1.0) & (interpretability_score_df['y_true'] == 1.0)]['score'].mean()
# def calculate_interpretability_scores(y_pred_for_thresholds, y_true_multivariate: np.ndarray, contribution_per_var_ranking: np.ndarray, max_k) -> Dict:
#     interpretability_score_df = pd.DataFrame(columns=['score'])
#     interpretability_score_df['y_true'] = (y_true_multivariate.sum(axis=1) >= 1).astype(float)
#     interpretability_score_df['true_anomalous_dimension_count'] = y_true_multivariate.sum(axis=1)
#
#     y_pred_overlapped_for_thresholds = np.apply_along_axis(lambda col: (col == 1) & (interpretability_score_df['y_true'] == 1), 0, y_pred_for_thresholds)
#
#     score_for_thresholds_df = pd.DataFrame(columns=[f'score_threshold_{i}' for i in range(y_pred_for_thresholds.shape[1])], data=np.zeros((y_true_multivariate.shape[0], y_pred_for_thresholds.shape[1])))
#     y_pred_for_thresholds_df = pd.DataFrame(columns=[f'y_pred_threshold_{i}' for i in range(y_pred_for_thresholds.shape[1])], data=y_pred_for_thresholds)
#     y_pred_overlapped_for_thresholds_df = pd.DataFrame(columns=[f'y_pred_overlapped_threshold_{i}' for i in range(y_pred_overlapped_for_thresholds.shape[1])], data=y_pred_overlapped_for_thresholds)
#
#     interpretability_score_df = pd.concat([interpretability_score_df, score_for_thresholds_df, y_pred_for_thresholds_df, y_pred_overlapped_for_thresholds_df], axis=1)
#     # interpretability_score_df[[f'score_threshold_{i}' for i in range(y_pred_for_thresholds.shape[1])]] = np.zeros(
#     #     (y_true_multivariate.shape[0], y_pred_for_thresholds.shape[1]))
#     # interpretability_score_df[
#     #     [f'y_pred_threshold_{i}' for i in range(y_pred_for_thresholds.shape[1])]] = y_pred_for_thresholds
#
#     # interpretability_score_df.loc[interpretability_score_df['y_pred'] == 1.0, 'score'] = interpretability_score_df.loc[interpretability_score_df['y_true'] == 1.0].loc[interpretability_score_df['y_pred'] == 1.0, 'detected_anomalous_dimension_count'] / interpretability_score_df.loc[interpretability_score_df['y_true'] == 1.0].loc[interpretability_score_df['y_pred'] == 1.0, 'true_anomalous_dimension_count']
#     # interpretability_scores = np.zeros(y_pred.shape[0])
#
#     results_for_all_k = dict()
#     start_time = time.time()
#     for top_k in range(1, max_k + 1):
#         # score_for_thresholds = []
#         top_k_indices_matrix = contribution_per_var_ranking[:, -top_k:]
#         interpretability_score_df[f'detected_anomalous_dimension_count_top_{top_k}'] = np.take_along_axis(y_true_multivariate, top_k_indices_matrix, axis=1).sum(axis=1)
#
#         interpretability_score_df[f'score_unconditional'] = interpretability_score_df.apply(lambda row: row[f'detected_anomalous_dimension_count_top_{top_k}'] / row['true_anomalous_dimension_count'] if row['true_anomalous_dimension_count'] >= 1.0 else np.nan, axis=1)
#         # score_for_thresholds = [interpretability_score_df[interpretability_score_df[f'y_pred_overlapped_threshold_{threshold_index}'] == 1]['score_unconditional'].mean() for threshold_index in range(y_pred_for_thresholds.shape[1])]
#         # score_for_thresholds = np.take_along_axis(interpretability_score_df['score_unconditional'].values[:, np.newaxis], np.where(y_pred_overlapped_for_thresholds_df.values == 1, np.arange(y_pred_for_thresholds.shape[1]), -1), axis=1).mean(axis=0)
#         # score_for_thresholds = np.nanmean(np.where(y_pred_overlapped_for_thresholds_df.values == 1, interpretability_score_df[[f'score_unconditional']].values, np.nan), axis=0)
#         # score_for_thresholds = np.where(np.isnan(score_for_thresholds), 0.0, score_for_thresholds)
#
#         score_for_thresholds_selected = np.where(y_pred_overlapped_for_thresholds_df.values == 1,
#                                                    interpretability_score_df[[f'score_unconditional']].values, np.nan)
#         # if np.all(np.isnan(score_for_thresholds_selected), axis=0).any():
#         #     warnings.warn(f"All interpretability scores for at least one threshold are NaN for top_k={top_k}. This means that for at least one threshold, there are no detected true anomalies. Setting the interpretability score for these thresholds to 0.0.")
#         # score_for_thresholds = np.where(np.all(np.isnan(score_for_thresholds_selected), axis=0), 0.0, np.nanmean(score_for_thresholds_selected, axis=0))
#
#         all_nan_threshold_indexes = np.where(np.all(np.isnan(score_for_thresholds_selected), axis=0))[0]
#         score_for_thresholds_selected[:, all_nan_threshold_indexes] = 0.0
#         score_for_thresholds = np.nanmean(score_for_thresholds_selected, axis=0)
#         # score_for_thresholds = np.where(np.isnan(score_for_thresholds), 0.0, score_for_thresholds)
#         results_for_all_k[top_k] = np.array(score_for_thresholds)
#         # for threshold_index in range(y_pred_for_thresholds.shape[1]):
#         #     detected_true_anomaly_index = interpretability_score_df[interpretability_score_df[f'y_pred_overlapped_threshold_{threshold_index}'] == 1].index
#         #     interpretability_score_df.loc[detected_true_anomaly_index, f'score_threshold_{threshold_index}'] = interpretability_score_df.loc[detected_true_anomaly_index, f'detected_anomalous_dimension_count_top_{top_k}'].values / interpretability_score_df.loc[detected_true_anomaly_index,'true_anomalous_dimension_count'].values
#         #     score_for_thresholds.append(interpretability_score_df[interpretability_score_df[f'y_pred_overlapped_threshold_{threshold_index}'] == 1][f'score_threshold_{threshold_index}'].mean())
#         #     results_for_all_k[top_k] = np.array(score_for_thresholds)
#     end_time = time.time()
#     # print(f"Time taken to calculate interpretability scores for all thresholds: {end_time - start_time:.2f} seconds")
#     return results_for_all_k
#     # detected_true_anomaly_index = interpretability_score_df[interpretability_score_df['y_pred'] == 1.0].loc[interpretability_score_df['y_true'] == 1.0].index
#     # interpretability_score_df.loc[detected_true_anomaly_index, 'score'] = interpretability_score_df.loc[detected_true_anomaly_index, 'detected_anomalous_dimension_count'].values / interpretability_score_df.loc[detected_true_anomaly_index,'true_anomalous_dimension_count'].values
#     # return interpretability_score_df[(interpretability_score_df['y_pred'] == 1.0) & (interpretability_score_df['y_true'] == 1.0)]['score'].mean()
#
# def calculate_unconditional_interpretability_scores(y_true_multivariate: np.ndarray, dimension_contribution_ranking: np.ndarray, top_k) -> np.ndarray:
#     top_k_indices_matrix = dimension_contribution_ranking[:, -top_k:]
#     interpretability_score_df = pd.DataFrame(columns=['score'])
#     interpretability_score_df['score'] = np.zeros(y_true_multivariate.shape[0])
#     interpretability_score_df['y_true'] = (y_true_multivariate.sum(axis=1) >= 1).astype(float)
#     interpretability_score_df['detected_anomalous_dimension_count'] = np.take_along_axis(y_true_multivariate, top_k_indices_matrix, axis=1).sum(axis=1)
#     interpretability_score_df['true_anomalous_dimension_count'] = y_true_multivariate.sum(axis=1)
#     detected_true_anomaly_index = interpretability_score_df[interpretability_score_df['y_true'] == 1.0].index
#     interpretability_score_df.loc[detected_true_anomaly_index, 'score'] = interpretability_score_df.loc[detected_true_anomaly_index, 'detected_anomalous_dimension_count'].values / interpretability_score_df.loc[detected_true_anomaly_index,'true_anomalous_dimension_count'].values
#     return interpretability_score_df[interpretability_score_df['y_true'] == 1.0]['score'].mean()
#
#
# # def convert_raw_anomaly_score_per_var_to_contribution_percentage(y_score_per_var):
# #     y_score_per_var[np.isnan(y_score_per_var)] = 0.0
# #     y_score_per_var = np.where(y_score_per_var < 0, 0, y_score_per_var)
# #     y_score_per_var += 0.001
# #     # y_score_per_var += 0.001
# #     y_score_per_var = y_score_per_var / y_score_per_var.sum(axis=1, keepdims=True)
# #     return y_score_per_var
# #
# # def distribution_distance(ground_truth: np.ndarray, prediction: np.ndarray) -> float:
# #     assert ground_truth.ndim == 1
# #     assert prediction.ndim == 1
# #
# #     return kl_div(ground_truth, prediction).sum()
# #
# #     # distance = 0
# #     #
# #     # for distribution_true, distribution_predict in zip(ground_truth, prediction):
# #     #     distance += -(distribution_true*np.log(distribution_predict) + abs(distribution_true-distribution_predict)*np.log(abs(distribution_true-distribution_predict)))
# #     # return distance
# #
# # class InterpretabilityLogScore:
# #     """Takes an anomaly scoring and ground truth labels to compute and apply a threshold to the scoring.
# #
# #     Subclasses of this abstract base class define different strategies to put a threshold over the anomaly scorings.
# #     All strategies produce binary labels (0 or 1; 1 for anomalous) in the form of an integer NumPy array.
# #     The strategy :class:`~timeeval.metrics.thresholding.NoThresholding` is a special no-op strategy that checks for
# #     already existing binary labels and keeps them untouched. This allows applying the metrics on existing binary
# #     classification results.
# #     """
# #
# #     def __init__(self, include_negative: bool) -> None:
# #         self.include_negative: Optional[int] = include_negative
# #
# #     def score(self, y_true_multivariate: np.ndarray, y_score_per_var: np.ndarray) -> None:
# #         assert y_true_multivariate.ndim == 2
# #         assert y_score_per_var.ndim == 2
# #         # fpr, tpr, thresholds = roc_curve(y_true.reshape(-1), y_score.reshape(-1))
# #         # result = auc(fpr, tpr)
# #
# #         y_true = (y_true_multivariate.sum(axis=1)>=1).astype(float)
# #
# #         # anomaly_scores_per_var_ranking = np.argsort(y_score_per_var, axis=1)
# #         # top_k_anomalous_dimension = anomaly_scores_per_var_ranking[:, -self.top_k:]
# #         interpretability_list = []
# #         # smooth = 0.1
# #
# #         # y_score_per_var[np.isnan(y_score_per_var)] = 0.0
# #         # y_score_per_var = np.where(y_score_per_var < 0, 0, y_score_per_var)
# #         y_score_per_var = convert_raw_anomaly_score_per_var_to_contribution_percentage(y_score_per_var)
# #         # y_score_per_var += 0.001
# #         # y_score_per_var = y_score_per_var/y_score_per_var.sum(axis=1, keepdims=True)
# #         y_true_multivariate = y_true_multivariate + 0.1
# #         y_true_multivariate = y_true_multivariate/ y_true_multivariate.sum(axis=1, keepdims=True)
# #         for labels, anomaly_score_per_var, aggregated_label in zip(y_true_multivariate, y_score_per_var, y_true):
# #             # labels = softmax(labels)
# #             # anomaly_score_per_var = softmax(anomaly_score_per_var)
# #             interpretability = distribution_distance(labels, anomaly_score_per_var)
# #             if interpretability == np.inf:
# #                 print('Debug')
# #             if self.include_negative:
# #                 interpretability_list.append(interpretability)
# #             else:
# #                 if aggregated_label != 0.0:
# #                     interpretability_list.append(interpretability)
# #                 else:
# #                     interpretability_list.append(0.0)
# #         # interpretability_scores = np.sqrt(np.power(multivariate_labels-anomaly_scores_per_var,2).sum(axis=1))
# #         interpretability_scores = np.array(interpretability_list)
# #
# #         return interpretability_scores[y_true == 1].mean()
# #
# #     # def supports_continuous_scorings(self) -> bool:
# #     #     return True
# #     @property
# #     def name(self) -> str:
# #         return f'Interpretability_Log_Score'.upper()
#
#
# class Segment:
#     def __init__(self, start: int, end: int):
#         self.start = start
#         self.end = end
#     def __json__(self):
#         return {"start": self.start, "end": self.end}
#     def __len__(self):
#         return self.end - self.start + 1
#
# # class InterpretabilityConditionalHitKScoreWithArea:
# #
# #     def __init__(self, max_k, slope=50) -> None:
# #         self.max_k: Optional[int] = max_k
# #
# #     @property
# #     def name(self)-> str:
# #         return 'Interpretability_Conditional_Hit_{}_Score_With_Area_No_K_Mix'.upper()
# #
# #     def score_and_output_details(self, y_true_univariate, y_score_univariate, y_true_multivariate: np.ndarray, y_contribution_multivariate: np.ndarray) -> Dict:
# #         assert y_true_multivariate.ndim == 2
# #         assert y_contribution_multivariate.ndim == 2
# #         assert y_true_univariate.ndim == 1
# #         assert y_score_univariate.ndim == 1
# #         # fpr, tpr, thresholds = roc_curve(y_true.reshape(-1), y_score.reshape(-1))
# #         # result = auc(fpr, tpr)
# #
# #         y_true = (y_true_multivariate.sum(axis=1)>=1).astype(float)
# #         assert np.all(y_true == y_true_univariate)
# #
# #         precision, recall, thresholds = precision_recall_curve(y_true_univariate, np.round(y_score_univariate,2))
# #         # print('precision:', precision.shape, ' recall:', recall.shape, ' thresholds:', thresholds.shape)
# #         metric_matrix = np.stack((precision[:-1], recall[:-1], thresholds), axis=1)
# #         precision_recal_sum = metric_matrix[:, 0] + metric_matrix[:, 1]
# #         metric_matrix = metric_matrix[precision_recal_sum != 0]
# #         precision = metric_matrix[:, 0]
# #         recall = metric_matrix[:, 1]
# #         thresholds = metric_matrix[:, 2]
# #         # detete_indices = []
# #         # for i, (p, r, t) in enumerate(zip(precision, recall, thresholds)):
# #         #     if p + r == 0:
# #         #         detete_indices.append(i)
# #         # precision = np.delete(precision, detete_indices, axis=0)
# #         # recall = np.delete(recall, detete_indices, axis=0)
# #         # thresholds = np.delete(thresholds, detete_indices, axis=0)
# #
# #         f1_scores = 2 * (precision * recall) / (precision + recall)
# #         max_f1_score = np.nanmax(f1_scores)
# #         optimal_threshold = thresholds[np.nanargmax(f1_scores)]
# #
# #         y_score_multivariate = y_contribution_multivariate * y_score_univariate[:, np.newaxis]
# #         # assert np.all(y_score_multivariate.sum(axis=1) == y_score_univariate)
# #
# #         interpretability_list_dict ={k: np.zeros(y_true_univariate.shape[0]) for k in range(1, self.max_k+1)}
# #         all_results_dict = dict({k:[] for k in range(1, self.max_k+1)})
# #         # all_results = []
# #         ground_truth_anomaly_segments = extract_anomalous_segments(y_true_univariate)
# #         for threshold in thresholds:
# #             # interpretability_list = np.zeros(y_true_univariate.shape[0])
# #             y_pred = np.array(y_score_univariate >= threshold, dtype=float)
# #             selected_anomaly_scores_per_dimension = np.where(np.tile((y_pred == 1.0).reshape(-1,1), y_score_multivariate.shape[1]), y_score_multivariate, 0.0)
# #             detected_anomalies = np.array(y_pred == y_true_univariate, dtype=float)
# #             detected_anomaly_segments = extract_anomalous_segments(detected_anomalies)
# #
# #             overlaping_segments_list = find_overlapping_segments(detected_anomaly_segments, ground_truth_anomaly_segments)
# #             for overlaping_segment, detected_segment, ground_truth_segment in overlaping_segments_list:
# #                 start, end = overlaping_segment.start, overlaping_segment.end
# #                 detected_start, detected_end = detected_segment.start, detected_segment.end
# #                 ground_truth_start, ground_truth_end = ground_truth_segment.start, ground_truth_segment.end
# #                 # calculate the contribution of each dimension for the detected segment
# #                 anomaly_scores_per_dimension = selected_anomaly_scores_per_dimension[start:end+1]
# #                 dimension_contribution_accumulated = np.trapz(anomaly_scores_per_dimension, axis=0)
# #                 # top_1_dimension_idx = dimension_contribution_accumulated.argmax()
# #                 # if np.any(y_true_multivariate[detected_start:detected_end + 1, top_1_dimension_idx] == 1.0):
# #                 #     interpretability_list_dict[1][detected_start:detected_end + 1] = 1.0
# #
# #                 dimension_ranking = np.argsort(dimension_contribution_accumulated)
# #                 ground_truth_anomalous_dimensions = y_true_multivariate[ground_truth_start:ground_truth_end + 1].sum(
# #                         axis=0) >= 1
# #                 number_of_anomalous_dimensions = ground_truth_anomalous_dimensions.sum()
# #                 # top_k_dimension_to_consider = dimension_ranking[-number_of_anomalous_dimensions:]
# #                 detected_anomalous_dimensions = y_true_multivariate[detected_start:detected_end + 1,
# #                                                 dimension_ranking[-number_of_anomalous_dimensions:]].sum(axis=0) >= 1
# #                 if np.any(detected_anomalous_dimensions):
# #                     interpretability_list_dict[1][detected_start:detected_end+1] = detected_anomalous_dimensions.sum() / number_of_anomalous_dimensions
# #                 # dimension_ranking = np.argsort(dimension_contribution_accumulated)
# #                 # ground_truth_anomalous_dimensions = y_true_multivariate[ground_truth_start:ground_truth_end + 1].sum(
# #                 #     axis=0) >= 1
# #                 # for k in range(1, self.max_k+1):
# #                 #     max_detected_anomalous_dimensions = min(ground_truth_anomalous_dimensions.sum(), k)
# #                 #     detected_anomalous_dimensions = y_true_multivariate[detected_start:detected_end+1, dimension_ranking[-k:]].sum(axis=0)>=1
# #                 #     if np.any(detected_anomalous_dimensions):
# #                 #         interpretability_list_dict[k][detected_start:detected_end+1] = 1.0
# #                 # if np.any(y_true_multivariate[detected_start:detected_end+1, top_1_dimension_idx] == 1.0):
# #                 #     interpretability_list[detected_start:detected_end+1] = 1.0
# #
# #             # true_possitives = (y_pred == 1.0) & (y_true_univariate == 1.0)
# #             true_groundtruth_positives = (y_true_univariate == 1.0)
# #             true_preds_positives = (y_pred == 1.0)
# #
# #
# #             if np.any(true_groundtruth_positives) == False or np.any(true_preds_positives) == False:
# #                 # warnings.warn(f'No true positives detected at threshold {threshold}. Interpretability score is set to 0.0 for this threshold.')
# #                 for key in interpretability_list_dict.keys():
# #                     all_results_dict[key].append((0.0, threshold))
# #             else:
# #                 if self.name.lower().endswith('no_k_ground_truth'):
# #                     for key in interpretability_list_dict.keys():
# #                         combined_score = interpretability_list_dict[key][true_groundtruth_positives].mean()
# #                         all_results_dict[key].append((combined_score, threshold))
# #                 elif self.name.lower().endswith('no_k_preds'):
# #                     for key in interpretability_list_dict.keys():
# #                         combined_score = interpretability_list_dict[key][true_preds_positives].mean()
# #                         all_results_dict[key].append((combined_score, threshold))
# #                 else:
# #                     for key in interpretability_list_dict.keys():
# #                         tmp1 = interpretability_list_dict[key][true_preds_positives].mean()
# #                         tmp2 = interpretability_list_dict[key][true_groundtruth_positives].mean()
# #                         if tmp1 + tmp2 == 0:
# #                             combined_score = 0.0
# #                         else:
# #                             combined_score =2 * tmp1 * tmp2/ (tmp1 + tmp2)
# #                         all_results_dict[key].append((combined_score, threshold))
# #
# #
# #                     # all_results_dict[key].append((interpretability_list_dict[key][true_groundtruth_possitives].mean(), threshold))
# #         all_results_dict = {k:np.array(all_results_dict[k]) for k in all_results_dict.keys()}
# #         all_results_dict = {k: np.trapz(all_results_dict[k][:, 0], all_results_dict[k][:, 1]) for k in all_results_dict.keys()}
# #         return all_results_dict
# class InterpretabilityConditionalHitKScoreWithArea:
#
#     def __init__(self, max_k, slope=50) -> None:
#         self.max_k: Optional[int] = max_k
#
#     @property
#     def name(self)-> str:
#         return 'Interpretability_Conditional_Hit_{}_Score_With_Area_No_K_Mix'.upper()
#
#     def score_and_output_details(self, y_true_univariate, y_score_univariate, y_true_multivariate: np.ndarray, y_contribution_multivariate: np.ndarray) -> Dict:
#         assert y_true_multivariate.ndim == 2
#         assert y_contribution_multivariate.ndim == 2
#         assert y_true_univariate.ndim == 1
#         assert y_score_univariate.ndim == 1
#         # fpr, tpr, thresholds = roc_curve(y_true.reshape(-1), y_score.reshape(-1))
#         # result = auc(fpr, tpr)
#
#         y_true = (y_true_multivariate.sum(axis=1)>=1).astype(float)
#         assert np.all(y_true == y_true_univariate)
#
#         results_dict = dict()
#
#         precision, recall, thresholds = precision_recall_curve(y_true_univariate, np.round(y_score_univariate,4))
#         f1_scores = 2 * recall * precision / (recall + precision)
#         best_th_ix = np.nanargmax(f1_scores)
#         optimal_threshold = thresholds[best_th_ix]
#         print('precision:', precision.shape, ' recall:', recall.shape, ' thresholds:', thresholds.shape)
#         metric_matrix = np.stack((precision[:-1], recall[:-1], thresholds, f1_scores[:-1]), axis=1)
#         precision_recal_sum = metric_matrix[:, 0] + metric_matrix[:, 1]
#         metric_matrix = metric_matrix[precision_recal_sum != 0]
#         precision = metric_matrix[:, 0]
#         recall = metric_matrix[:, 1]
#         thresholds = metric_matrix[:, 2]
#         f1_scores = metric_matrix[:, 3]
#         results_dict['precision'] = precision
#         results_dict['recall'] = recall
#
#         results_dict['thresholds'] = thresholds
#         results_dict['f1_scores'] = f1_scores
#         results_dict['best_f1_score'] = f1_scores[best_th_ix]
#         results_dict['best_f1_idx'] = best_th_ix
#         results_dict['optimal_threshold'] = optimal_threshold
#
#
#
#         y_score_multivariate = y_contribution_multivariate * y_score_univariate[:, np.newaxis]
#         # assert np.all(y_score_multivariate.sum(axis=1) == y_score_univariate)
#
#         interpretability_list_dict ={k: np.zeros(y_true_univariate.shape[0]) for k in range(1, self.max_k+1)}
#         all_results_dict = dict()
#         for note in ['no_k_preds','no_k_ground_truth','no_k_mix']:
#             all_results_dict[note] = {k: [] for k in range(1, self.max_k + 1)}
#         # all_results = []
#         ground_truth_anomaly_segments = extract_anomalous_segments(y_true_univariate)
#         results_dict['count_groundtruth_anomalies'] = np.zeros(thresholds.shape)
#         results_dict['count_predicted_anomalies'] = np.zeros(thresholds.shape)
#         results_dict['count_overlapped_anomalies'] = np.zeros(thresholds.shape)
#         for threshold_index, threshold in enumerate(thresholds):
#             # interpretability_list = np.zeros(y_true_univariate.shape[0])
#             y_pred = np.array(y_score_univariate >= threshold, dtype=float)
#             selected_anomaly_scores_per_dimension = np.where(np.tile((y_pred == 1.0).reshape(-1,1), y_score_multivariate.shape[1]), y_score_multivariate, 0.0)
#             detected_anomalies = np.array(y_pred == y_true_univariate, dtype=float)
#             detected_anomaly_segments = extract_anomalous_segments(detected_anomalies)
#
#             overlaping_segments_list = find_overlapping_segments(detected_anomaly_segments, ground_truth_anomaly_segments)
#             for overlaping_segment, detected_segment, ground_truth_segment in overlaping_segments_list:
#                 start, end = overlaping_segment.start, overlaping_segment.end
#                 detected_start, detected_end = detected_segment.start, detected_segment.end
#                 ground_truth_start, ground_truth_end = ground_truth_segment.start, ground_truth_segment.end
#                 # calculate the contribution of each dimension for the detected segment
#                 anomaly_scores_per_dimension = selected_anomaly_scores_per_dimension[start:end+1]
#                 dimension_contribution_accumulated = np.trapz(anomaly_scores_per_dimension, axis=0)
#                 # top_1_dimension_idx = dimension_contribution_accumulated.argmax()
#                 # if np.any(y_true_multivariate[detected_start:detected_end + 1, top_1_dimension_idx] == 1.0):
#                 #     interpretability_list_dict[1][detected_start:detected_end + 1] = 1.0
#
#                 dimension_ranking = np.argsort(dimension_contribution_accumulated)
#                 ground_truth_anomalous_dimensions = y_true_multivariate[ground_truth_start:ground_truth_end + 1].sum(
#                         axis=0) >= 1
#                 number_of_anomalous_dimensions = ground_truth_anomalous_dimensions.sum()
#                 # top_k_dimension_to_consider = dimension_ranking[-number_of_anomalous_dimensions:]
#                 detected_anomalous_dimensions = y_true_multivariate[detected_start:detected_end + 1,
#                                                 dimension_ranking[-number_of_anomalous_dimensions:]].sum(axis=0) >= 1
#                 if np.any(detected_anomalous_dimensions):
#                     interpretability_list_dict[1][detected_start:detected_end+1] = detected_anomalous_dimensions.sum() / number_of_anomalous_dimensions
#                 # dimension_ranking = np.argsort(dimension_contribution_accumulated)
#                 # ground_truth_anomalous_dimensions = y_true_multivariate[ground_truth_start:ground_truth_end + 1].sum(
#                 #     axis=0) >= 1
#                 # for k in range(1, self.max_k+1):
#                 #     max_detected_anomalous_dimensions = min(ground_truth_anomalous_dimensions.sum(), k)
#                 #     detected_anomalous_dimensions = y_true_multivariate[detected_start:detected_end+1, dimension_ranking[-k:]].sum(axis=0)>=1
#                 #     if np.any(detected_anomalous_dimensions):
#                 #         interpretability_list_dict[k][detected_start:detected_end+1] = 1.0
#                 # if np.any(y_true_multivariate[detected_start:detected_end+1, top_1_dimension_idx] == 1.0):
#                 #     interpretability_list[detected_start:detected_end+1] = 1.0
#
#             # true_possitives = (y_pred == 1.0) & (y_true_univariate == 1.0)
#
#             true_preds_positives = (y_pred == 1.0)
#             true_groundtruth_positives = (y_true_univariate == 1.0)
#             true_overlaped_possitives = (y_pred == 1.0) & (y_true_univariate == 1)
#             results_dict['count_overlapped_anomalies'][threshold_index] = true_overlaped_possitives.sum().astype(np.float32)
#             results_dict['count_predicted_anomalies'][threshold_index] = true_preds_positives.sum().astype(np.float32)
#             results_dict['count_groundtruth_anomalies'][threshold_index] = true_groundtruth_positives.sum().astype(np.float32)
#
#
#
#
#             if np.any(true_overlaped_possitives) == False:
#                 # warnings.warn(f'No true positives detected at threshold {threshold}. Interpretability score is set to 0.0 for this threshold.')
#                 for key in interpretability_list_dict.keys():
#                     all_results_dict['no_k_preds'][key].append((0.0, threshold, true_preds_positives.astype(np.float32).sum()))
#                     all_results_dict['no_k_ground_truth'][key].append((0.0, threshold, true_groundtruth_positives.astype(np.float32).sum()))
#                     all_results_dict['no_k_mix'][key].append((0.0, threshold, true_overlaped_possitives.astype(np.float32).sum()))
#             else:
#                 # if self.name.lower().endswith('no_k_ground_truth'):
#                 for key in interpretability_list_dict.keys():
#                     combined_score = interpretability_list_dict[key][true_overlaped_possitives].mean()
#                     all_results_dict['no_k_ground_truth'][key].append((combined_score, threshold, true_groundtruth_positives.astype(np.float32).sum()))
#                 # elif self.name.lower().endswith('no_k_preds'):
#                 for key in interpretability_list_dict.keys():
#                     combined_score = interpretability_list_dict[key][true_overlaped_possitives].mean()
#                     all_results_dict['no_k_preds'][key].append((combined_score, threshold, true_preds_positives.astype(np.float32).sum()))
#                 # else:
#                 for key in interpretability_list_dict.keys():
#                     tmp1 = interpretability_list_dict[key][true_preds_positives].mean()
#                     tmp2 = interpretability_list_dict[key][true_groundtruth_positives].mean()
#                     if tmp1 + tmp2 == 0:
#                         combined_score = 0.0
#                     else:
#                         combined_score =2 * tmp1 * tmp2/ (tmp1 + tmp2)
#                     all_results_dict['no_k_mix'][key].append((combined_score, threshold, true_overlaped_possitives.astype(np.float32).sum()))
#                     # all_results_dict[key].append((interpretability_list_dict[key][true_groundtruth_possitives].mean(), threshold))
#
#         for note in ['no_k_preds','no_k_ground_truth','no_k_mix']:
#             all_results_dict[note] = {k: np.array(all_results_dict[note][k]) for k in all_results_dict[note].keys()}
#
#         # all_results_dict = {k: np.trapz(all_results_dict[k][:, 0], all_results_dict[k][:, 1]) for k in all_results_dict.keys()}
#         results_dict['interpretability_metrics'] = all_results_dict
#         return results_dict
#
# def extract_anomalous_segments(labels: np.array) -> List[Segment]:
#     """
#     Extract contiguous anomalous segments from binary timestamp labels.
#
#     Args:
#         labels: list of length T with values {0,1} (1 = anomaly)
#
#     Returns:
#         List of (start_idx, end_idx) pairs, inclusive indices.
#     """
#     segments: List[Segment] = []
#     in_seg = False
#     start = 0
#
#     for i, y in enumerate(labels):
#         if y == 1 and not in_seg:
#             in_seg = True
#             start = i
#         elif y == 0 and in_seg:
#             segments.append(Segment(start, i - 1))
#             in_seg = False
#
#     # handle segment that ends at the last timestamp
#     if in_seg:
#         segments.append(Segment(start, len(labels) - 1))
#
#     return segments
# def find_overlapping_segments(detected_segments: List[Segment], ground_truth_segments: List[Segment]) -> List[Tuple[Segment, Segment, Segment]]:
#     """
#     Find overlapping segments between two lists of segments.
#
#     Args:
#         detected_segments: list of (start_idx, end_idx) pairs
#         ground_truth_segments: list of (start_idx, end_idx) pairs
#
#     Returns:
#         List of (start_idx, end_idx) pairs representing the overlapping segments.
#     """
#     overlapping_segments = []
#     for detected_segment in detected_segments:
#         start_a, end_a = detected_segment.start, detected_segment.end
#         for ground_truth_segment in ground_truth_segments:
#             start_b, end_b = ground_truth_segment.start, ground_truth_segment.end
#             overlap_start = max(start_a, start_b)
#             overlap_end = min(end_a, end_b)
#             if overlap_start <= overlap_end:
#                 overlapping_segments.append((Segment(overlap_start, overlap_end), Segment(start_a, end_a), Segment(start_b, end_b)))
#     return overlapping_segments
#
# def estimate_dimension_contribution_with_a_buffer(dimension_contribution: np.ndarray, buffer: int) -> np.ndarray:
#     """
#     Estimate the contribution of each dimension with a buffer.
#
#     Args:
#         dimension_contribution: 2D array of shape (T, D) with contribution scores for each dimension at each timestamp.
#         buffer: number of timestamps to consider before and after the detected anomaly.
#
#     Returns:
#         2D array of shape (T, D) with estimated contribution scores for each dimension at each timestamp.
#     """
#     T, D = dimension_contribution.shape
#     estimated_contribution = np.zeros_like(dimension_contribution)
#
#     for t in range(T):
#         start = max(0, t - buffer)
#         end = min(T, t + buffer + 1)
#         estimated_contribution[t] = np.trapz(dimension_contribution[start:end, :], axis=0)
#
#     return estimated_contribution
#
# def calculate_max_f1_scores_with_interpretability(precision, recall):
#     precision_recall_sum = precision + recall
#     precision = precision[precision_recall_sum != 0]
#     recall = recall[precision_recall_sum != 0]
#     f1_scores = 2 * (precision * recall) / (precision + recall)
#     if len(f1_scores) == 0:
#         return 0.0
#     return np.max(f1_scores)