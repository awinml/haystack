import logging

from typing import Union, List, Dict, Any
from haystack.preview import Document, component, Pipeline, default_from_dict, default_to_dict
from haystack.preview.evaluation import EvaluationResult
from haystack.preview.lazy_imports import LazyImport


def eval(
    runnable: Union[Pipeline, component], inputs: List[Dict[str, Any]], expected_outputs: List[Dict[str, Any]]
) -> EvaluationResult:
    outputs = []

    # Check that expected outputs has the correct shape
    if len(inputs) != len(expected_outputs):
        raise ValueError("Length of expected_ouputs does not match length of inputs.")

    for input_ in inputs:
        output = runnable.run(input_)
        outputs.append(output)

    return EvaluationResult(runnable, inputs, outputs, expected_outputs)
