from typing import Optional

import numpy as np

from streamauc.utils import AggregationMethod, check_confusion_matrix_entries

__all__ = ["accuracy"]


def accuracy(
    tp: np.ndarray,
    tn: np.ndarray,
    fp: np.ndarray,
    fn: np.ndarray,
    method: AggregationMethod = AggregationMethod.MACRO,
    class_index: Optional[int] = None,
    check_inputs: bool = True,
    **kwargs,
):
    if method == AggregationMethod.MICRO:
        _acc = accuracy_micro(
            tp=tp, tn=tn, fn=fn, fp=fp, check_inputs=check_inputs
        )
    elif method == AggregationMethod.MACRO:
        _acc = accuracy_macro(
            tp=tp, fn=fn, fp=fp, tn=tn, check_inputs=check_inputs
        )
    elif method == AggregationMethod.ONE_VS_ALL:
        _acc = accuracy_onevsall(
            tp=tp, fn=fn, fp=fp, tn=tn, check_inputs=check_inputs
        )[..., class_index]
    else:
        raise ValueError(
            f"Method must one of {[e.value for e in AggregationMethod]}. "
            f"Got {method}."
        )
    return _acc


def accuracy_onevsall(
    tp: np.ndarray,
    tn: np.ndarray,
    fn: np.ndarray,
    fp: np.ndarray,
    check_inputs: bool = True,
):
    if check_inputs:
        check_confusion_matrix_entries(fn, tp, fp)

    _total = tp + tn + fp + fn
    _accuracy = (tp + tn) / (_total + 1e-12)
    return _accuracy


def accuracy_micro(
    tp: np.ndarray,
    tn: np.ndarray,
    fn: np.ndarray,
    fp: np.ndarray,
    check_inputs: bool = True,
):
    if check_inputs:
        check_confusion_matrix_entries(fn, tp, fp)

    tp_sum = np.sum(tp, axis=-1)
    tn_sum = np.sum(tn, axis=-1)
    fn_sum = np.sum(fn, axis=-1)
    fp_sum = np.sum(fp, axis=-1)

    return accuracy_onevsall(
        tp=tp_sum, tn=tn_sum, fn=fn_sum, fp=fp_sum, check_inputs=check_inputs
    )


def accuracy_macro(
    tp: np.ndarray,
    tn: np.ndarray,
    fn: np.ndarray,
    fp: np.ndarray,
    check_inputs: bool = True,
):
    _f1_onevsall = accuracy_onevsall(
        tp=tp, tn=tn, fn=fn, fp=fp, check_inputs=check_inputs
    )
    return np.mean(_f1_onevsall, axis=-1)
