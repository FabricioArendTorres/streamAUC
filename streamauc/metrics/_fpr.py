from typing import Optional

import numpy as np

from streamauc.utils import AggregationMethod, check_confusion_matrix_entries

__all__ = ["fpr"]


def fpr(
    fp: np.ndarray,
    tn: np.ndarray,
    method: AggregationMethod = AggregationMethod.MACRO,
    class_index: Optional[int] = None,
    check_inputs: bool = True,
    **kwargs,
) -> np.ndarray:
    """
    Compute the false positive rate (FPR) given the false positive (fp) and
    true negative (tn) predictions at various thresholds.
    Can be used as a Callable for the auc method.

    Parameters
    ----------
    fp : np.ndarray
        Array of false positives for each class.
        Of shape [num_thresholds, num_classes]
    tn : np.ndarray
        Array of true negatives for each class.
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
    fpr : np.ndarray
        FPR for the specified class across different samples.
        Of shape [num_thresholds]

    Raises
    ------
    ValueError
        If an invalid aggregation method is specified.

    Notes
    -----
    - For micro-averaging:
      $$ \\text{FPR}_{\\text{micro}} =
      \\frac{\\sum \\text{FP}}{\\sum (\\text{FP} + \\text{TN})} $$
    - For macro-averaging:
      $$ \\text{FPR}_{\\text{macro}} =
      \\frac{1}{C} \\sum_{c=1}^{C} \\frac{\\text{FP}_c}{\\text{FP}_c +
      \\text{TN}_c} $$
    - For one-vs-all:
      $$ \\text{FPR}_{\\text{one\\_vs\\_all}} =
      \\frac{\\text{FP}_{c}}{\\text{FP}_{c} + \\text{TN}_{c}} $$
      where $ c $ is the specified class index.
    """
    if method == AggregationMethod.MICRO:
        _fpr = fpr_micro(fp=fp, tn=tn, check_inputs=check_inputs)
    elif method == AggregationMethod.MACRO:
        _fpr = fpr_macro(fp=fp, tn=tn, check_inputs=check_inputs)
    elif method == AggregationMethod.ONE_VS_ALL:
        _fpr = fpr_onevsall(fp=fp, tn=tn, check_inputs=check_inputs)[
            ..., class_index
        ]
    else:
        raise ValueError(
            f"Method must one of {[e.value for e in AggregationMethod]}. "
            f"Got {method}."
        )
    return _fpr


def fpr_macro(
    fp: np.ndarray, tn: np.ndarray, check_inputs: bool = True
) -> np.ndarray:
    """
    Compute macro-averaged false-positive rate (FPR) for multi-class
    classification.

    Parameters
    ----------
    fp : np.ndarray
        Array of false positives for each class.
        Of shape [num_thresholds, num_classes]
    tn : np.ndarray
        Array of true negatives for each class.
        Of shape [num_thresholds, num_classes]
    check_inputs : bool, optional
        If True, perform input validation checks. Default is True.

    Returns
    -------
    fpr : np.ndarray
        Macro-averaged false positive rate for each class.
        Of shape [num_thresholds]

    Notes
    -----
    - FPR (False Positive Rate) is calculated as the mean of individual
    class FPRs:
      $$ \\text{FPR} = \\frac{1}{C} \\sum_{c=1}^{C}
      \\frac{\\text{FP}_c}{\\text{FP}_c + \\text{TN}_c} $$
    """

    if check_inputs:
        check_confusion_matrix_entries(fp, tn)

    _fpr = fpr_onevsall(fp=fp, tn=tn, check_inputs=check_inputs).mean(axis=-1)
    return _fpr


def fpr_micro(
    fp: np.ndarray, tn: np.ndarray, check_inputs: bool = True
) -> np.ndarray:
    """
    Compute micro-averaged false positive rate (FPR)
    for multi-class classification.

    Parameters
    ----------
    fp : np.ndarray
        Array of false positives for each class.
        Of shape [num_thresholds, num_classes]
    tn : np.ndarray
        Array of true negatives for each class.
        Of shape [num_thresholds, num_classes]
    check_inputs : bool, optional
        If True, perform input validation checks. Default is True.

    Returns
    -------
    fpr : np.ndarray
        Micro-averaged false positive rate for each class.
        Of shape [num_thresholds]

    Notes
    -----
    - FPR (False Positive Rate) is calculated as the ratio of false positives
     to the sum of false positives and true negatives:
      $$ \\text{FPR} = \\frac{\\sum_i \\text{FP}_i}{\\sum_i (\\text{FP}_i +
      \\text{TN}_i)} $$
    """
    if check_inputs:
        check_confusion_matrix_entries(fp, tn)

    fp_sum = np.sum(fp, axis=-1)
    tn_sum = np.sum(tn, axis=-1)

    _fpr = fp_sum / (fp_sum + tn_sum + 1e-12)
    return _fpr


def fpr_onevsall(
    fp: np.ndarray,
    tn: np.ndarray,
    check_inputs: bool = True,
) -> np.ndarray:
    """
    Compute TPR and FPR for a one-vs-all multi-class classification setup.

    Parameters
    ----------
    fp : np.ndarray
        Array of false positives for each class.
        Of shape [num_thresholds, num_classes]
    tn : np.ndarray
        Array of true negatives for each class.
        Of shape [num_thresholds, num_classes]
    check_inputs : bool, optional
        If True, perform input validation checks. Default is True.

    Returns
    -------
    fpr : np.ndarray
        False positive rate for each class across different samples.
        Of shape [num_thresholds, num_classes]

    Notes
    -----
    - FPR (False Positive Rate) is calculated as the ratio of false positives
     to the
      sum of false positives and true negatives for the specified class:
      $$ \\text{FPR} =
      \\frac{\\text{FP}_{c}}{\\text{FP}_{c} + \\text{TN}_{c}} $$
    """
    if check_inputs:
        check_confusion_matrix_entries(fp, tn)

    _fpr = fp / (fp + tn + 1e-12)
    return _fpr
