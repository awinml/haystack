from typing import Dict, Callable, Union
from pathlib import Path

from enum import Enum

# MetricsResult = Dict[str, Dict[str, float]]
# MetricCalculator = Callable[..., MetricsResult]


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
        # Dump info to file here
        pass


class EvaluationResult:
    def __init__(self, runnable, inputs, outputs, expected_outputs):
        self.runnable = runnable
        self.inputs = inputs
        self.outputs = outputs
        self.expected_outputs = expected_outputs

    def calculate_metrics(self, metric: Union[Metric, Callable[..., MetricsResult]], **kwargs) -> MetricsResult:
        # Verify if we're calculating a known metric
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

        # If it's not a known metric it must be a custom one
        return metric(self, **kwargs)

    def _calculate_recall(self):
        pass

    def _calculate_mrr(self):
        pass

    def _calculate_accuracy(self):
        pass

    def _calculate_map(self):
        pass

    def _calculate_f1(self):
        pass

    def _calculate_em(self):
        pass

    def _calculate_sas(self):
        pass
