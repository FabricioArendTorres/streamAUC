from typing import Optional

import numpy as np

from streamauc.metrics._fpr import fpr
from streamauc.utils import AggregationMethod

__all__ = ["tnr"]


def tnr(
    fp: np.ndarray,
    tn: np.ndarray,
    method: AggregationMethod = AggregationMethod.MACRO,
    class_index: Optional[int] = None,
    check_inputs: bool = True,
    **kwargs,
) -> np.ndarray:
    """
    Compute the true negative rate (TNR) given the false positive (fp) and
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
    method : AggregationMethod, optional
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
    tnr : np.ndarray
        TNR for the specified class across different samples.
        Of shape [num_thresholds]

    Raises
    ------
    ValueError
        If an invalid aggregation method is specified.

    Notes
    -----
    - For micro-averaging:
      $$ \\text{TNR}_{\\text{micro}} = 1 -
      \\frac{\\sum \\text{FP}}{\\sum (\\text{FP} + \\text{TN})} $$
    - For macro-averaging:
      $$ \\text{TNR}_{\\text{macro}} = 1 -
      \\frac{1}{C} \\sum_{c=1}^{C} \\frac{\\text{FP}_c}{\\text{FP}_c +
       \\text{TN}_c} $$
    - For one-vs-all:
      $$ \\text{TNR}_{\\text{one\\_vs\\_all}} = 1 -
      \\frac{\\text{FP}_{c}}{\\text{FP}_{c} + \\text{TN}_{c}} $$
      where $ c $ is the specified class index.
    """
    return 1 - fpr(
        fp=fp,
        tn=tn,
        method=method,
        class_index=class_index,
        check_inputs=check_inputs,
        **kwargs,
    )
