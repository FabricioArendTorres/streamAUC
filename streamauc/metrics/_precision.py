from typing import Optional

import numpy as np

from streamauc.utils import AggregationMethod, check_confusion_matrix_entries

__all__ = ["precision"]


def precision(
    tp: np.ndarray,
    fp: np.ndarray,
    method: AggregationMethod = AggregationMethod.MACRO,
    class_index: Optional[int] = None,
    check_inputs: bool = True,
    **kwargs,
):
    """
    Compute precision for multi-class classification using the
    specified aggregation method.

    Parameters
    ----------
    tp : np.ndarray
        Array of true positives for each class.
    fp : np.ndarray
        Array of false positives for each class.
    method : AggregationMethod, optional
        Method used to compute precision for multiple classes.
        Default is AggregationMethod.MACRO.
        Must be one of ["macro", "micro", "one_vs_all"].
    class_index : int, optional
        Class index for "one_vs_all" calculation. Required if `method`
        is "one_vs_all".
    check_inputs : bool, optional
        If True, perform input validation checks. Default is True.
    **kwargs
        Additional keyword arguments.

    Returns
    -------
    precision : np.ndarray
        Computed precision values based on the specified aggregation method.

    Raises
    ------
    ValueError
        If an invalid aggregation method is specified.

    Notes
    -----
    - For micro-averaging:
      $$ \\text{Precision}_{\\text{micro}} =
      \\frac{\\sum \\text{TP}}{\\sum (\\text{TP} + \\text{FP})} $$
    - For macro-averaging:
      $$ \\text{Precision}_{\\text{macro}} =
      \\frac{1}{C} \\sum_{c=1}^{C}
      \\frac{\\text{TP}_c}{\\text{TP}_c + \\text{FP}_c} $$
    - For one-vs-all:
      $$ \\text{Precision}_{\\text{one\\_vs\\_all}} =
      \\frac{\\text{TP}_{c}}{\\text{TP}_{c} + \\text{FP}_{c}} $$
      where $ c $ is the specified class index.
    """
    if method == AggregationMethod.MICRO:
        precision = precision_micro(tp=tp, fp=fp, check_inputs=check_inputs)
    elif method == AggregationMethod.MACRO:
        precision = precision_macro(tp=tp, fp=fp, check_inputs=check_inputs)
    elif method == AggregationMethod.ONE_VS_ALL:
        precision = precision_onevsall(
            tp=tp, fp=fp, check_inputs=check_inputs
        )[..., class_index]
    else:
        raise ValueError(
            f"Method must one of {[e.value for e in AggregationMethod]}. "
            f"Got {method}."
        )

    return precision


def precision_onevsall(
    tp: np.ndarray, fp: np.ndarray, check_inputs: bool = True
):
    """
    Compute precision and recall for a one-vs-all multi-class classification
     setup for all classes.

    Parameters
    ----------
    tp : np.ndarray
        Array of true positives for each class.
        Of shape [num_thresholds, num_classes]
    fp : np.ndarray
        Array of false positives for each class.
        Of shape [num_thresholds, num_classes]
    class_index : int
        Index of the class for which to compute precision and recall.
    check_inputs : bool, optional
        If True, perform input validation checks. Default is True.

    Returns
    -------
    precision : np.ndarray
        Precision for the specified class across different samples.
        Of shape [num_thresholds, num_classes]

    Notes
    -----
    - Precision is calculated as the ratio of true positives to the sum of true
      positives and false positives for the specified class:
      $$ \\text{Precision} = \\frac{\\text{TP}_{c}}{\\text{TP}_{c} +
      \\text{FP}_{c}} $$
    """
    if check_inputs:
        check_confusion_matrix_entries(fp, tp)

    precision = tp / (tp + fp + 1e-10)
    return precision


def precision_micro(
    tp: np.ndarray, fp: np.ndarray, check_inputs: bool = True
) -> np.ndarray:
    """
    Compute micro-averaged precision for multi-class classification.

    Parameters
    ----------
    tp : np.ndarray
        Array of true positives for each threshold and class.
        Of shape [num_thresholds, num_classes]
    fp : np.ndarray
        Array of false positives for each threshold and class.
        Of shape [num_thresholds, num_classes]
    check_inputs : bool
        Whether to check inputs with asserts.

    Returns
    -------
    precision : np.ndarray
        Micro-averaged precision for each threshold.
        Of shape [num_thresholds]

    Notes
    -----
    - Precision is calculated as the ratio of true positives to the sum of true
      positives and false positives:
      $$ \\text{Precision} = \\frac{\\sum_i \\text{TP}_i}{\\sum_i (\\text{
      TP}_i + \\text{FP}_i)} $$
    """

    if check_inputs:
        check_confusion_matrix_entries(fp, tp)

    tp_sum = np.sum(tp, axis=-1)
    fp_sum = np.sum(fp, axis=-1)
    return precision_onevsall(tp=tp_sum, fp=fp_sum, check_inputs=check_inputs)


def precision_macro(
    tp: np.ndarray, fp: np.ndarray, check_inputs: bool = True
) -> np.ndarray:
    """
    Compute macro-averaged precision and recall for multi-class classification.

    Parameters
    ----------
    tp : np.ndarray
        Array of true positives for each class.
        Of shape [num_thresholds, num_classes]
    fp : np.ndarray
        Array of false positives for each class.
        Of shape [num_thresholds, num_classes]
    check_inputs : bool, optional
        If True, perform input validation checks. Default is True.

    Returns
    -------
    precision : np.ndarray
        Macro-averaged precision for each class.
        Of shape [num_thresholds]

    Notes
    -----
    - Precision is calculated as the mean of individual class precisions:
      $$ \\text{Precision} = \\frac{1}{C} \\sum_{c=1}^{C}
      \\frac{\\text{TP}_c}{\\text{TP}_c + \\text{FP}_c} $$
    """
    if check_inputs:
        check_confusion_matrix_entries(fp, tp)

    precision = precision_onevsall(
        tp=tp, fp=fp, check_inputs=check_inputs
    ).mean(axis=-1)
    return precision
