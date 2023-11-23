import logging

from typing import Union, List, Dict, Any
from haystack.preview import Document, component, Pipeline, default_from_dict, default_to_dict
from haystack.preview.evaluation import EvaluationResult
from haystack.preview.lazy_imports import LazyImport


logger = logging.getLogger(__name__)


with LazyImport() as modeling_import:
    from haystack.modeling.evaluation.metrics import semantic_answer_similarity
    from haystack.modeling.evaluation.squad import compute_f1 as calculate_f1_str
    from haystack.modeling.evaluation.squad import compute_exact as calculate_em_str
    from haystack.utils.context_matching import calculate_context_similarity


def eval(
    runnable: Union[Pipeline, component], inputs: List[Dict[str, Any]], expected_outputs: List[Dict[str, Any]]
) -> EvaluationResult:
    outputs = []

    if len(inputs) != len(expected_outputs):
        raise ValueError("Length of expected_ouputs does not match length of inputs.")

    for input_ in inputs:
        output = runnable.run(input_)
        outputs.append(output)
    return EvaluationResult(runnable, inputs, outputs, expected_outputs)
