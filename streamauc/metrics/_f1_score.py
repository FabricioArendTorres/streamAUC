from typing import Optional

import numpy as np

from streamauc.utils import AggregationMethod, check_confusion_matrix_entries

__all__ = ["f1_score"]


def f1_score(
    tp: np.ndarray,
    fp: np.ndarray,
    fn: np.ndarray,
    method: AggregationMethod = AggregationMethod.MACRO,
    class_index: Optional[int] = None,
    check_inputs: bool = True,
    **kwargs,
):
    if method == AggregationMethod.MICRO:
        _f1 = f1_micro(tp=tp, fn=fn, fp=fp, check_inputs=check_inputs)
    elif method == AggregationMethod.MACRO:
        _f1 = f1_macro(tp=tp, fn=fn, fp=fp, check_inputs=check_inputs)
    elif method == AggregationMethod.ONE_VS_ALL:
        _f1 = f1_onevsall(tp=tp, fn=fn, fp=fp, check_inputs=check_inputs)[
            ..., class_index
        ]
    else:
        raise ValueError(
            f"Method must one of {[e.value for e in AggregationMethod]}. "
            f"Got {method}."
        )
    return _f1


def f1_onevsall(
    tp: np.ndarray, fn: np.ndarray, fp: np.ndarray, check_inputs: bool = True
):
    if check_inputs:
        check_confusion_matrix_entries(fn, tp, fp)

    _f1 = (2 * tp) / (2 * tp + fp + fn + 1e-12)
    return _f1


def f1_micro(
    tp: np.ndarray, fn: np.ndarray, fp: np.ndarray, check_inputs: bool = True
):
    if check_inputs:
        check_confusion_matrix_entries(fn, tp, fp)

    tp_sum = np.sum(tp, axis=-1)
    fn_sum = np.sum(fn, axis=-1)
    fp_sum = np.sum(fp, axis=-1)

    return f1_onevsall(
        tp=tp_sum, fn=fn_sum, fp=fp_sum, check_inputs=check_inputs
    )


def f1_macro(
    tp: np.ndarray, fn: np.ndarray, fp: np.ndarray, check_inputs: bool = True
):
    _f1_onevsall = f1_onevsall(tp=tp, fn=fn, fp=fp, check_inputs=check_inputs)
    return np.mean(_f1_onevsall, axis=-1)
