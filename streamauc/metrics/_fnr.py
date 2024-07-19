from typing import Optional

import numpy as np

from streamauc.metrics import tpr
from streamauc.utils import AggregationMethod

__all__ = ["fnr"]


def fnr(
    tp: np.ndarray,
    fn: np.ndarray,
    method: AggregationMethod = AggregationMethod.MACRO,
    class_index: Optional[int] = None,
    check_inputs: bool = True,
    **kwargs
):
    """
    Compute the false negative rate (FNR) given the true positive (tp) and
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
    fnr : np.ndarray
        FNR for the specified class across different samples.
        Of shape [num_thresholds]

    Raises
    ------
    ValueError
        If an invalid aggregation method is specified.

    Notes
    -----
    - For micro-averaging:
      $$ \\text{FNR}_{\\text{micro}} = 1 - \\frac{\\sum \\text{TP}}{
      \\sum (\\text{TP} + \\text{FN})} $$
    - For macro-averaging:
      $$ \\text{FNR}_{\\text{macro}} = 1 - \\frac{1}{C} \\sum_{c=1}^{C}
      \\frac{\\text{TP}_c}{\\text{TP}_c + \\text{FN}_c} $$
    - For one-vs-all:
      $$ \\text{FNR}_{\\text{one\\_vs\\_all}} = 1 -
       \\frac{\\text{TP}_{c}}{\\text{TP}_{c} + \\text{FN}_{c}} $$
      where $ c $ is the specified class index.
    """
    return 1 - tpr(
        tp=tp,
        fn=fn,
        method=method,
        class_index=class_index,
        check_inputs=check_inputs,
        **kwargs
    )
