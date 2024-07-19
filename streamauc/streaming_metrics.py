from typing import Callable, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

import streamauc.metrics as metrics
from streamauc.plot_util import plot_curve_and_auc
from streamauc.utils import AggregationMethod, auc

__all__ = ["StreamingMetrics"]


def _validate_thresholds(
    num_thresholds: int,
    thresholds: Optional[Union[List[float], np.ndarray]] = None,
) -> Tuple[int, np.ndarray]:
    if thresholds is not None:
        if (
            np.min(thresholds) < 0
            or np.max(thresholds) > 1
            or len(thresholds) <= 1
        ):
            raise ValueError(
                f"Values must be in range [0., 1.],"
                f" found values in range "
                f"[{np.min(thresholds)}, {np.max(thresholds)}]"
            )
        thresholds = np.sort(thresholds)
        num_thresholds = len(thresholds) + 2
    else:
        if num_thresholds <= 1:
            raise ValueError(
                "Argument `num_thresholds` must be an integer > 1. "
                f"Received: num_thresholds={num_thresholds}"
            )
        thresholds = np.linspace(0, 1, num_thresholds - 2)

    # Add endpoints slightly below 0 and above 1 to account for floating point
    # imprecisions
    epsilon = np.finfo(float).eps
    thresholds = np.concatenate(
        ([0.0 - epsilon], thresholds, [1.0 + epsilon])
    )[::-1]

    return num_thresholds, thresholds


class StreamingMetrics:
    """
    Class for keeping track of metrics for many thresholds in a
    minibatch-wise, iterative, fashion.

    Parameters
    ----------
    num_thresholds : int, optional
        Number of thresholds to evaluate the curve. Default is 200.
    num_classes : int
        Number of classes in the multiclass setting. Must be >= 2.
    thresholds : list of float, optional
        List of specific thresholds to evaluate the metrics at.
        A probability >= threshold is defined as a positive prediction for
        the respective class.
    """

    def __init__(
        self,
        num_thresholds: int = 200,
        num_classes: int = 2,
        thresholds: Optional[Union[List[float], np.ndarray]] = None,
    ):
        if num_classes < 2:
            raise ValueError("Argument `num_classes` must be an integer >= 2.")

        self.num_classes = num_classes

        self.num_thresholds, self.thresholds = _validate_thresholds(
            num_thresholds, thresholds
        )

        self._confusion_matrix = np.zeros(
            (self.num_thresholds, self.num_classes, 2, 2),
            dtype=int,
        )

    @property
    def confusion_matrix(self) -> np.ndarray:
        """
        For each threshold, and for each class, there is a 2x2 confusion
        matrix. The entries of each confusion matrix correspond to the
        labels of:
        np.array([ ["TP","FN"],
                   ["FP", "TN"]
                   ])

        That is, the indices are given by:
        TP: self.confusion_matrix[..., 0,0]
        FP: self.confusion_matrix[..., 1,0]
        FN: self.confusion_matrix[..., 0,1]
        TN: self.confusion_matrix[..., 1,1]

        Note, that this corresponds to the flipped confusion matrix of
        sklearn. We prefer this order, as it corresponds to many written
        references, e.g. the wikipedia page.

        That is, self.confusion_matrix = np.flip(sklearn_confusion_matrix)

        """
        return self._confusion_matrix

    def reset(self):
        """
        Reset the intermediate values for the confusion matrix.
        """
        self._confusion_matrix = np.zeros(
            (self.num_thresholds, self.num_classes, 2, 2),
            dtype=int,
        )

    def update(
        self,
        y_true: np.ndarray,
        y_score: np.ndarray,
        check_inputs: bool = True,
    ):
        """
        Update the intermediate values based on streaming data.

        Parameters
        ----------
        y_true : np.ndarray
            Ground truth labels of shape [-1] (or [-1, 1], [-1, 1, 1, 1]...)
            with values indicating the class index. Alternatively, may also
            be one-hot encoded labels of shape [-1, num_classes].
        y_score : np.ndarray
            Predicted probabilities for each class of shape[-1, num_classes]

        Raises
        ------
        ValueError
            If the shapes of `y_true` and `y_pred` do not match.
        """

        y_true = np.squeeze(y_true).astype(int)
        y_score = np.squeeze(y_score)

        if check_inputs:
            if y_true.ndim > 2:
                raise ValueError(
                    f"Unknown shape of y_true: {y_true.shape},"
                    f"must be squeezable to either [-1, num_classes] or [-1]."
                )
            if y_true.ndim == 2 and np.any(y_true.sum(-1) != 1):
                raise ValueError("The provided one-hot encoding is invalid.")
            if y_score.ndim > 2:
                raise ValueError(
                    f"Unknown shape of y_true: {y_true.shape},"
                    f"must be squeezable to either [-1, num_classes] or [-1]."
                )

            if not (y_true.shape[0] == y_score.shape[0]):
                raise ValueError(
                    "Number of samples in y_true and y_pred must match"
                )

            if (y_score.ndim == 2) and (y_score.shape[1] != self.num_classes):
                raise ValueError(f"Invalid shape of y_pred: {y_score.shape}")

        if y_true.ndim == 2 and y_true.shape[1] == self.num_classes:
            y_onehot = y_true
        else:
            y_onehot = np.eye(self.num_classes, dtype=int)[y_true]

        # use numpy broadcasting to get predictions
        pred_pos = y_score[np.newaxis, ...] >= self.thresholds.reshape(
            -1, 1, 1
        )
        is_pos = y_onehot[np.newaxis, ...]

        # sum over the minibatch samples
        tp = np.sum(pred_pos & is_pos, 1)
        fp = np.sum(pred_pos & (~is_pos), 1)
        fn = np.sum((~pred_pos) & (is_pos), 1)
        tn = np.sum((~pred_pos) & (~is_pos), 1)

        # update confusion matrix entry
        self._confusion_matrix[..., 0, 0] += tp
        self._confusion_matrix[..., 1, 0] += fp
        self._confusion_matrix[..., 1, 1] += tn
        self._confusion_matrix[..., 0, 1] += fn

    def _total(self) -> np.ndarray:
        """
        Calculate total for each threshold and class.
        Of course, this should be the same value for all thresholds.

        Returns
        -------
        np.ndarray
            Total at each threshold.
        """
        total = self.confusion_matrix.sum(-1).sum(-1)
        return total

    def true_positives(self) -> np.ndarray:
        """
        Calculate true positives for each threshold and class.

        Returns
        -------
        np.ndarray
            True positives at each threshold.
        """

        return self.confusion_matrix[..., 0, 0]
        # tp = np.diagonal(self._confusion_matrix, axis1=1, axis2=2)
        # return tp

    def false_positives(self) -> np.ndarray:
        """
        Calculate false positives for each threshold and class.

        Returns
        -------
        np.ndarray
            False positives at each threshold.
        """
        # tp = self.true_positives()
        # pp = self.predicted_positives()
        #
        # fp = pp - tp
        # return fp

        return self.confusion_matrix[..., 1, 0]

    def true_negatives(self) -> np.ndarray:
        """
        Calculate the total negatives for each threshold and class.

        Returns
        -------
        np.ndarray
            Negatives at each threshold.
        """

        return self.confusion_matrix[..., 1, 1]

    def false_negatives(self) -> np.ndarray:
        """
        Calculate false negatives for each threshold and class.

        Returns
        -------
        np.ndarray
            False negatives at each threshold.
        """
        # tp = self.true_positives()
        # p = self.positives()
        #
        # fn = p - tp
        return self.confusion_matrix[..., 0, 1]

    def positives(self) -> np.ndarray:
        """
        Calculate the total positives for each threshold and class.

        Returns
        -------
        np.ndarray
            Positives at each threshold.
        """
        # return np.sum(self._confusion_matrix, axis=-1)
        return (
            self.confusion_matrix[..., 0, 0] + self.confusion_matrix[..., 0, 1]
        )

    def negatives(self) -> np.ndarray:
        """
        Calculate the total negatives for each threshold and class.

        Returns
        -------
        np.ndarray
            Negatives at each threshold.
        """

        return self._total() - self.positives()

    def predicted_positives(self):
        """
        Calculate predicted positives for each threshold and class.

        Returns
        -------
        np.ndarray
            Predicted positives at each threshold.
        """

        return (
            self.confusion_matrix[..., 0, 0] + self.confusion_matrix[..., 1, 0]
        )

    def predicted_negatives(self) -> np.ndarray:
        """
        Calculate predicted negatives for each threshold and class.

        Returns
        -------
        np.ndarray
            Predicted positives at each threshold.
        """
        #
        # pp = self.predicted_positives()
        # total = self._total()
        #
        # pn = total - pp
        # return pn

        return (
            self.confusion_matrix[..., 0, 1] + self.confusion_matrix[..., 1, 1]
        )

    def calc_metric(
        self,
        metric: Callable,
        method: AggregationMethod = AggregationMethod.MACRO,
        class_index: Optional[int] = None,
        check_inputs=True,
    ):
        tp = self.true_positives()
        fp = self.false_positives()
        fn = self.false_negatives()
        tn = self.true_negatives()

        return metric(
            tp=tp,
            fp=fp,
            fn=fn,
            tn=tn,
            class_index=class_index,
            method=method,
            check_inputs=check_inputs,
        )

    def auc(
        self,
        metric_xaxis: Callable = metrics.fpr,
        metric_yaxis: Callable = metrics.tpr,
        method: AggregationMethod = AggregationMethod.ONE_VS_ALL,
        class_index: Optional[int] = None,
        check_inputs=True,
    ):

        metric_args = dict(
            tp=self.true_positives(),
            fp=self.false_positives(),
            fn=self.false_negatives(),
            tn=self.true_negatives(),
            class_index=class_index,
            method=method,
            check_inputs=check_inputs,
        )

        x_values = metric_xaxis(**metric_args)
        y_values = metric_yaxis(**metric_args)
        return auc(x_values, y_values)

    def precision_recall_curve(
        self,
        method: AggregationMethod = AggregationMethod.ONE_VS_ALL,
        class_index: Optional[int] = None,
        check_inputs: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute precision and recall at all thresholds for plotting and auc
        computation. We adopt the behaviour of sklearn, in that the
        precision corresponding to a recall of 0 is 1.

        (Technically its undefined since its tp/(tp+fp) with tp=fp=0,
        but the value of 1 serves for stable plotting.)

        Parameters
        ----------
        method : AggregationMethod
            Method used to compute precision and recall for multiple classes.
            Micro and macro refer to the averaging method.
            Macro computes the metric for each class, and then averages the
            metrics.
            If '1-vs-all' the index for the positive class has to be defined
            in 'class_index'. All other classes will be summarized as the
            negative class.

            Must be one of ["macro","micro","1-vs-all"].

        class_index : int, optional
            Class index for "1-vs-all" calculation.
             Required if `method` is "1-vs-all".

        Returns
        -------
        precision : np.ndarray
            Precision values at each threshold.
        recall : np.ndarray
            Recall values at each threshold.
        """
        tp = self.true_positives()
        fp = self.false_positives()
        fn = self.false_negatives()

        precision = metrics.precision(
            tp=tp,
            fp=fp,
            method=method,
            class_index=class_index,
            check_inputs=check_inputs,
        )

        recall = metrics.recall(
            tp=tp,
            fn=fn,
            method=method,
            class_index=class_index,
            check_inputs=check_inputs,
        )
        # ensure precision 1 at recall 0
        # precision[0] = 1.0
        return (
            precision[::-1][1:].squeeze(),
            recall[::-1][1:].squeeze(),
            self.thresholds[1:][::-1].squeeze(),
        )

    def roc_curve(
        self,
        method: AggregationMethod = AggregationMethod.ONE_VS_ALL,
        class_index: Optional[int] = None,
        check_inputs: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        tp = self.true_positives()
        fp = self.false_positives()
        fn = self.false_negatives()
        tn = self.true_negatives()

        _tpr = metrics.tpr(
            tp=tp,
            fn=fn,
            method=method,
            class_index=class_index,
            check_inputs=check_inputs,
        )
        _fpr = metrics.fpr(
            fp=fp,
            tn=tn,
            method=method,
            class_index=class_index,
            check_inputs=check_inputs,
        )
        return _fpr.squeeze(), _tpr.squeeze(), self.thresholds.squeeze()

    def plot_roc_curve(
        self,
        class_names: Optional[List[str]] = None,
        method=AggregationMethod.ONE_VS_ALL,
        class_index: Optional[int] = None,
        **kwargs,
    ) -> plt.Figure:  #
        # pragma: nocover
        fpr, tpr, thresholds = self.roc_curve(
            method=method,
            class_index=class_index,
        )

        if method != AggregationMethod.ONE_VS_ALL:
            class_names = None
        return plot_curve_and_auc(
            x_values=fpr,
            y_values=tpr,
            thresholds=thresholds,
            class_names=class_names,
            **kwargs,
        )

    def plot_precision_recall_curve(
        self,
        class_names: Optional[List[str]] = None,
        method=AggregationMethod.ONE_VS_ALL,
        class_index: Optional[int] = None,
        **kwargs,
    ) -> plt.Figure:  #

        # pragma: nocover
        precision, recall, thresholds = self.precision_recall_curve(
            method=method,
            class_index=class_index,
        )

        if method != AggregationMethod.ONE_VS_ALL:
            class_names = None
            assert (
                class_index is None
            ), "class_index is only usable for ONE_VS_ALL"
        return plot_curve_and_auc(
            x_values=recall,
            y_values=precision,
            thresholds=thresholds,
            class_names=class_names,
            **kwargs,
        )
