# pragma: no cover
from typing import Optional

import numpy as np

from streamauc.metrics._f1_score import f1_score
from streamauc.metrics._fpr import fpr
from streamauc.metrics._precision import precision
from streamauc.metrics._tnr import tnr
from streamauc.metrics._tpr import tpr
from streamauc.utils import AggregationMethod, copy_docstring_from

__all__ = [
    "recall",
    "sensitivity",
    "specificity",
    "selectivity",
    "positive_predictive_value",
    "dice",
    "hit_rate",
    "fallout",
]


# ###### TPR
@copy_docstring_from(tpr)
def recall(
    tp: np.ndarray,
    fn: np.ndarray,
    method: AggregationMethod = AggregationMethod.MACRO,
    class_index: Optional[int] = None,
    check_inputs: bool = True,
    **kwargs,
):
    return tpr(tp, fn, method, class_index, check_inputs, **kwargs)


@copy_docstring_from(tpr)
def sensitivity(
    tp: np.ndarray,
    fn: np.ndarray,
    method: AggregationMethod = AggregationMethod.MACRO,
    class_index: Optional[int] = None,
    check_inputs: bool = True,
    **kwargs,
):
    return tpr(tp, fn, method, class_index, check_inputs, **kwargs)


@copy_docstring_from(tpr)
def hit_rate(
    tp: np.ndarray,
    fn: np.ndarray,
    method: AggregationMethod = AggregationMethod.MACRO,
    class_index: Optional[int] = None,
    check_inputs: bool = True,
    **kwargs,
):
    return tpr(tp, fn, method, class_index, check_inputs, **kwargs)


# ###### fpr
@copy_docstring_from(fpr)
def fallout(
    fp: np.ndarray,
    tn: np.ndarray,
    method: AggregationMethod = AggregationMethod.MACRO,
    class_index: Optional[int] = None,
    check_inputs: bool = True,
    **kwargs,
):
    return fpr(fp, tn, method, class_index, check_inputs, **kwargs)


# ###### precision
@copy_docstring_from(precision)
def positive_predictive_value(
    tp: np.ndarray,
    fp: np.ndarray,
    method: AggregationMethod = AggregationMethod.MACRO,
    class_index: Optional[int] = None,
    check_inputs: bool = True,
    **kwargs,
):
    precision(tp, fp, method, class_index, check_inputs, **kwargs)


# ###### tnr
@copy_docstring_from(tnr)
def specificity(
    fp: np.ndarray,
    tn: np.ndarray,
    method: AggregationMethod = AggregationMethod.MACRO,
    class_index: Optional[int] = None,
    check_inputs: bool = True,
    **kwargs,
):
    return tnr(fp, tn, method, class_index, check_inputs, **kwargs)


@copy_docstring_from(tnr)
def selectivity(
    fp: np.ndarray,
    tn: np.ndarray,
    method: AggregationMethod = AggregationMethod.MACRO,
    class_index: Optional[int] = None,
    check_inputs: bool = True,
    **kwargs,
):
    return tnr(fp, tn, method, class_index, check_inputs, **kwargs)


# ###### f1
@copy_docstring_from(f1_score)
def dice(
    tp: np.ndarray,
    fp: np.ndarray,
    fn: np.ndarray,
    method: AggregationMethod = AggregationMethod.MACRO,
    class_index: Optional[int] = None,
    check_inputs: bool = True,
    **kwargs,
):
    return f1_score(
        tp=tp,
        fp=fp,
        fn=fn,
        method=method,
        class_index=class_index,
        check_inputs=check_inputs,
        **kwargs,
    )
