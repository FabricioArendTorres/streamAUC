from typing import Optional, Tuple

import numpy as np

from streamauc.utils import AggregationMethod, check_confusion_matrix_entries

__all__ = ["tpr"]


def tpr(
    tp: np.ndarray,
    fn: np.ndarray,
    method: AggregationMethod = AggregationMethod.MACRO,
    class_index: Optional[int] = None,
    check_inputs: bool = True,
    **kwargs,
):
    """
    Compute the true positive rate (TPR) given the true positive (tp) and
    false negative (fn) predictions at various thresholds.
    Can be used as a Callable for the auc method.

    Parameters
    ----------
    tp : np.ndarray
        Array of true positives for each class.
        Of shape [num_thresholds, num_classes]
    fn : np.ndarray
        Array of false negatives for each class.
        Of shape [num_thresholds, num_classes]
    method : AggregationMethod
        Aggregation method to be used in multiclass setting.
        Default is AggregationMethod.MACRO.
    class_index : int, optional
        Class index for "one_vs_all" calculation. Required if `method`
        is "one_vs_all".
    check_inputs : bool, optional
        If True, perform input validation checks. Default is True.
    **kwargs
        Additional keyword arguments.

    Returns
    -------
    tpr : np.ndarray
        TPR for the specified class across different samples.
        Of shape [num_thresholds]

    Raises
    ------
    ValueError
        If an invalid aggregation method is specified.

    Notes
    -----
    - For micro-averaging:
      $$ \\text{TPR}_{\\text{micro}} = \\frac{\\sum \\text{TP}}{
      \\sum (\\text{TP} + \\text{FN})} $$
    - For macro-averaging:
      $$ \\text{TPR}_{\\text{macro}} = \\frac{1}{C} \\sum_{c=1}^{C}
      \\frac{\\text{TP}_c}{\\text{TP}_c + \\text{FN}_c} $$
    - For one-vs-all:
      $$ \\text{TPR}_{\\text{one\\_vs\\_all}} = \\frac{\\text{TP}_{c}}{
      \\text{TP}_{c} + \\text{FN}_{c}} $$
      where $ c $ is the specified class index.
    """

    if method == AggregationMethod.MICRO:
        _tpr = tpr_micro(tp=tp, fn=fn, check_inputs=check_inputs)
    elif method == AggregationMethod.MACRO:
        _tpr = tpr_macro(tp=tp, fn=fn, check_inputs=check_inputs)
    elif method == AggregationMethod.ONE_VS_ALL:
        _tpr = tpr_onevsall(tp=tp, fn=fn, check_inputs=check_inputs)[
            ..., class_index
        ]
    else:
        raise ValueError(
            f"Method must one of {[e.value for e in AggregationMethod]}. "
            f"Got {method}."
        )
    return _tpr


def tpr_onevsall(tp: np.ndarray, fn: np.ndarray, check_inputs: bool = True):
    """
    Compute the true positive rate (TPR) for a one-vs-all multi-class
    classification setup for all classes.

    Parameters
    ----------
    tp : np.ndarray
        Array of true positives for each class.
        Of shape [num_thresholds, num_classes]
    fn : np.ndarray
        Array of false negatives for each class.
        Of shape [num_thresholds, num_classes]
    check_inputs : bool, optional
        If True, perform input validation checks. Default is True.

    Returns
    -------
    tpr : np.ndarray
        TPR for the specified class across different samples.
        Of shape [num_thresholds, num_classes]

    Notes
    -----
    - TPR is calculated as the ratio of true positives to the sum of true
      positives and false negatives for the specified class:
      $$ \\text{TPR} = \\frac{\\text{TP}_{c}}{\\text{TP}_{c} +
      \\text{FN}_{c}} $$
    """
    if check_inputs:
        check_confusion_matrix_entries(fn, tp)

    _tpr = tp / (tp + fn + 1e-12)
    return _tpr


def tpr_micro(
    tp: np.ndarray, fn: np.ndarray, check_inputs: bool = True
) -> np.ndarray:
    """
    Compute micro-averaged recall for multi-class classification.

    Parameters
    ----------
    tp : np.ndarray
        Array of true positives for each threshold and class.
        Of shape [num_thresholds, num_classes]
    fn : np.ndarray
        Array of false negatives for each threshold and class.
        Of shape [num_thresholds, num_classes]
    check_inputs : bool
        Whether to check inputs with asserts.

    Returns
    -------
    recall : np.ndarray
        Micro-averaged recall for each threshold.
        Of shape [num_thresholds]

    Notes
    -----
    - Recall is calculated as the ratio of true positives to the sum of true
      positives and false negatives:
      $$ \\text{Recall} = \\frac{\\sum_i \\text{TP}_i}{\\sum_i
      (\\text{TP}_i + \\text{FN}_i)} $$
    """

    if check_inputs:
        check_confusion_matrix_entries(fn, tp)

    tp_sum = np.sum(tp, axis=-1)
    fn_sum = np.sum(fn, axis=-1)
    _tpr = tp_sum / (tp_sum + fn_sum + 1e-12)
    return _tpr


def tpr_macro(
    tp: np.ndarray, fn: np.ndarray, check_inputs: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute macro-averaged TPR for multi-class classification.

    Parameters
    ----------
    tp : np.ndarray
        Array of true positives for each class.
        Of shape [num_thresholds, num_classes]
    fn : np.ndarray
        Array of false negatives for each class.
        Of shape [num_thresholds, num_classes]
    check_inputs : bool, optional
        If True, perform input validation checks. Default is True.

    Returns
    -------
    tpr : np.ndarray
        Macro-averaged true positive rate for each class.
        Of shape [num_thresholds]

    Notes
    -----
    - TPR (True Positive Rate) is calculated as the mean of individual
    class TPRs:
      $$ \\text{TPR} = \\frac{1}{C} \\sum_{c=1}^{C}
      \\frac{\\text{TP}_c}{\\text{TP}_c + \\text{FN}_c} $$
    """

    if check_inputs:
        check_confusion_matrix_entries(tp, fn)

    _tpr = (tp / (tp + fn + 1e-12)).mean(axis=-1)
    return _tpr
