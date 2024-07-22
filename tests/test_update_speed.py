import os
import unittest

import numpy as np

from streamauc import AggregationMethod, StreamingMetrics, auc
from streamauc.streaming_metrics import compute_confusion_matrix
from streamauc.utils import onehot_encode
import time


class TestConfusionMatrixUpdatesSpeed(unittest.TestCase):
    def setUp(self):
        np.random.seed(1234)

    def test_reset(self):
        thresholds = np.linspace(0, 1, 50)
        curve = StreamingMetrics(
            thresholds=thresholds,
            num_classes=5,
        )

        n_samples = 256 * 256 * 8
        num_repeats = 10  # Number of times to repeat the update call
        times = []

        for _ in range(num_repeats):
            y_true = np.random.randint(0, curve.num_classes - 1, n_samples)
            y_pred = np.random.random((n_samples, curve.num_classes))
            y_pred = y_pred / y_pred.sum(-1, keepdims=True)

            start_time = time.perf_counter()
            curve.update(y_true=y_true, y_score=y_pred)
            end_time = time.perf_counter()

            times.append(end_time - start_time)

        mean_time_s = np.mean(times)
        stdev_time_s = np.std(times)
        self.assertLess(mean_time_s, 1)
        self.assertLess(stdev_time_s / mean_time_s, 0.5)


class TestConfusionMatrixUpdatesCorrect(unittest.TestCase):
    def setUp(self):
        np.random.seed(1234)

    def test_reset(self):
        thresholds = np.linspace(0, 1, 50)
        curve = StreamingMetrics(
            thresholds=thresholds,
            num_classes=5,
        )

        n_samples = 1_000

        y_true = np.random.randint(0, curve.num_classes - 1, n_samples)
        y_onehot = onehot_encode(y_true, curve.num_classes)

        y_score = np.random.random((n_samples, curve.num_classes))
        y_score = y_score / y_score.sum(-1, keepdims=True)

        pred_pos = y_score[np.newaxis, ...] >= curve.thresholds.reshape(
            -1, 1, 1
        )
        is_pos = y_onehot[np.newaxis, ...]

        # sum over the minibatch samples
        ref_tp = np.sum(pred_pos & is_pos, 1)
        ref_fp = np.sum(pred_pos & (~is_pos), 1)
        ref_fn = np.sum((~pred_pos) & (is_pos), 1)
        ref_tn = np.sum((~pred_pos) & (~is_pos), 1)

        tp, fp, fn, tn = compute_confusion_matrix.py_func(
            pred_pos=pred_pos, y_onehot=y_onehot
        )

        np.testing.assert_allclose(ref_tp, tp)
        np.testing.assert_allclose(ref_fp, fp)
        np.testing.assert_allclose(ref_tn, tn)
        np.testing.assert_allclose(ref_fn, fn)


if __name__ == "__main__":
    unittest.main()
