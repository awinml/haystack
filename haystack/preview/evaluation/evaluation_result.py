from typing import Dict, Callable, Union, List
from pathlib import Path
import json

from copy import deepcopy
import numpy as np

from enum import Enum

from haystack.preview.evaluation.utils import group_values, get_grouped_values


class Metric(Enum):
    ACCURACY = "Accuracy"
    RECALL = "Recall"
    MRR = "Mean Reciprocal Rank"
    MAP = "Mean Average Precision"
    EM = "Exact Match"
    F1 = "F1"
    SAS = "SemanticAnswerSimilarity"


class MetricsResult(dict):
    def save(self, file: Union[str, Path]):
        with open(file, "w") as outfile:
            json.dump(self, outfile, indent=4)


class EvaluationResult:
    def __init__(self, runnable, inputs, outputs, expected_outputs):
        self.runnable = runnable
        self.inputs = inputs
        self.outputs = outputs
        self.expected_outputs = expected_outputs

        self.grouped_outputs = group_values(self.outputs)
        self.grouped_expected_outputs = group_values(self.expected_outputs)

        self.preds = get_grouped_values(self.grouped_outputs, "replies")
        self.labels = get_grouped_values(self.grouped_expected_outputs, "replies")

    def calculate_metrics(
        self, metrics: Union[List[Metric], List[Callable[..., MetricsResult]]], **kwargs
    ) -> MetricsResult:
        results = MetricsResult()

        for metric in metrics:
            if isinstance(metric, Metric):
                # Calculate standard metrics
                result = self._calculate_standard_metric(metric, **kwargs)
            else:
                # If it's not a known metric it must be a custom one
                result = metric(self, **kwargs)

            if isinstance(result, MetricsResult):
                results.update(result)

        return results

    def _calculate_standard_metric(self, metric: Metric, **kwargs) -> MetricsResult:
        if metric == Metric.RECALL:
            return self._calculate_recall(**kwargs)
        elif metric == Metric.MRR:
            return self._calculate_mrr(**kwargs)
        elif metric == Metric.MAP:
            return self._calculate_map(**kwargs)
        elif metric == Metric.ACCURACY:
            return self._calculate_accuracy(**kwargs)
        elif metric == Metric.F1:
            return self._calculate_f1(**kwargs)
        elif metric == Metric.EM:
            return self._calculate_em(**kwargs)
        elif metric == Metric.SAS:
            return self._calculate_sas(**kwargs)
        else:
            raise ValueError(f"Unknown metric: {metric}")

    def _calculate_accuracy(self):
        preds = self.preds
        labels = self.labels
        if type(preds) is type(labels) is list:
            preds = np.array(list(flatten_list(preds)))
            labels = np.array(list(flatten_list(labels)))
        assert type(preds) is type(labels) is np.ndarray
        correct = preds == labels
        return MetricsResult({"acc": correct.mean()})

    def _calculate_recall(self):
        preds = self.preds
        labels = self.labels
        if type(preds) is type(labels) is list:
            preds = np.array(list(flatten_list(preds)))
            labels = np.array(list(flatten_list(labels)))
        assert type(preds) is type(labels) is np.ndarray
        correct = preds == labels
        return MetricsResult({"recall": correct.mean()})

    def _calculate_mrr(self):
        pass

    def _calculate_map(self):
        pass

    def _calculate_f1(self):
        pass

    def _calculate_em(self):
        pass

    def _calculate_sas(self):
        pass


def flatten_list(nested_list):
    """Flatten an arbitrarily nested list, without recursion (to avoid
    stack overflows). Returns a new list, the original list is unchanged.
    >> list(flatten_list([1, 2, 3, [4], [], [[[[[[[[[5]]]]]]]]]]))
    [1, 2, 3, 4, 5]
    >> list(flatten_list([[1, 2], 3]))
    [1, 2, 3]
    """
    nested_list = deepcopy(nested_list)

    while nested_list:
        sublist = nested_list.pop(0)

        if isinstance(sublist, list):
            nested_list = sublist + nested_list
        else:
            yield sublist
