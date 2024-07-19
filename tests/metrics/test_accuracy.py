import unittest
import numpy as np
from unittest.mock import patch
from streamauc.metrics._accuracy import (
    accuracy,
    accuracy_micro,
    accuracy_macro,
    accuracy_onevsall,
)
from streamauc.utils import AggregationMethod


class TestAccuracyFunction(unittest.TestCase):
    @patch("streamauc.metrics._accuracy.accuracy_micro")
    def test_accuracy_micro(self, mock_acc_micro):
        tp = np.array([[10, 5, 0], [7, 8, 9]])
        tn = np.array([[10, 5, 0], [7, 8, 9]])
        fp = np.array([[2, 1, 0], [1, 0, 1]])
        fn = np.array([[1, 0, 2], [0, 2, 1]])

        mock_acc_micro.return_value = np.array([0.83333333, 0.92307692])

        result = accuracy(
            tp, tn, fp, fn, method=AggregationMethod.MICRO, check_inputs=True
        )

        mock_acc_micro.assert_called_once_with(
            tp=tp, tn=tn, fn=fn, fp=fp, check_inputs=True
        )

        np.testing.assert_almost_equal(result, mock_acc_micro.return_value)

    @patch("streamauc.metrics._accuracy.accuracy_macro")
    def test_accuracy_macro(self, mock_acc_macro):
        tp = np.array([[10, 5, 0], [7, 8, 9]])
        tn = np.array([[10, 5, 0], [7, 8, 9]])
        fp = np.array([[2, 1, 0], [1, 0, 1]])
        fn = np.array([[1, 0, 2], [0, 2, 1]])

        mock_acc_macro.return_value = np.array([0.83333333, 0.88888889])

        result = accuracy(
            tp, tn, fp, fn, method=AggregationMethod.MACRO, check_inputs=True
        )

        mock_acc_macro.assert_called_once_with(
            tp=tp, tn=tn, fn=fn, fp=fp, check_inputs=True
        )

        np.testing.assert_almost_equal(result, mock_acc_macro.return_value)

    @patch("streamauc.metrics._accuracy.accuracy_onevsall")
    def test_accuracy_onevsall(self, mock_acc_onevsall):
        tp = np.array([[10, 5, 0], [7, 8, 9]])
        tn = np.array([[10, 5, 0], [7, 8, 9]])
        fp = np.array([[2, 1, 0], [1, 0, 1]])
        fn = np.array([[1, 0, 2], [0, 2, 1]])
        class_index = 1

        mock_acc_onevsall.return_value = np.array(
            [[0.83333333, 0.92307692], [0.8, 0.9]]
        )

        result = accuracy(
            tp,
            tn,
            fp,
            fn,
            method=AggregationMethod.ONE_VS_ALL,
            class_index=class_index,
            check_inputs=True,
        )

        mock_acc_onevsall.assert_called_once_with(
            tp=tp, tn=tn, fn=fn, fp=fp, check_inputs=True
        )

        expected_acc = mock_acc_onevsall.return_value[..., class_index]
        np.testing.assert_almost_equal(result, expected_acc)

    def test_invalid_method(self):
        tp = np.array([[10, 5, 0], [7, 8, 9]])
        tn = np.array([[10, 5, 0], [7, 8, 9]])
        fp = np.array([[2, 1, 0], [1, 0, 1]])
        fn = np.array([[1, 0, 2], [0, 2, 1]])

        with self.assertRaises(ValueError):
            accuracy(
                tp, tn, fp, fn, method="invalid_method", check_inputs=True
            )


class TestAccuracyOneVsAll(unittest.TestCase):
    def test_accuracy_onevsall(self):
        tp = np.array([[10, 5, 0], [7, 8, 9]])
        tn = np.array([[10, 5, 0], [7, 8, 9]])
        fp = np.array([[2, 1, 0], [1, 0, 1]])
        fn = np.array([[1, 0, 2], [0, 2, 1]])

        expected_acc = (tp + tn) / (tp + tn + fp + fn)

        result = accuracy_onevsall(
            tp=tp, tn=tn, fn=fn, fp=fp, check_inputs=True
        )
        np.testing.assert_almost_equal(result, expected_acc)

    def test_zero_division_onevsall(self):
        # Test case with zeros to ensure no division by zero errors
        tp = np.array([[0, 0, 0], [0, 0, 0]])
        fp = np.array([[0, 0, 0], [0, 0, 0]])
        fn = np.array([[0, 0, 0], [0, 0, 0]])
        tn = np.array([[0, 0, 0], [0, 0, 0]])

        expected_acc = np.array([0.0, 0.0])
        result = accuracy_onevsall(
            tp=tp, tn=tn, fn=fn, fp=fp, check_inputs=True
        )
        self.assertEqual(result.shape, tp.shape)

        np.testing.assert_almost_equal(result[:, 0], expected_acc)
        np.testing.assert_almost_equal(result[:, 1], expected_acc)


class TestAccuracyMacro(unittest.TestCase):
    def test_accuracy_macro(self):
        tp = np.array([[10, 5, 0], [7, 8, 9]])
        tn = np.array([[10, 5, 0], [7, 8, 9]])
        fp = np.array([[2, 1, 0], [1, 0, 1]])
        fn = np.array([[1, 0, 2], [0, 2, 1]])

        expected_acc_onevsall = (tp + tn) / (tp + tn + fp + fn)
        expected_acc_macro = np.mean(expected_acc_onevsall, axis=-1)

        result = accuracy_macro(tp=tp, tn=tn, fn=fn, fp=fp, check_inputs=True)

        np.testing.assert_almost_equal(result, expected_acc_macro)

    def test_zero_division_macro(self):
        # Test case with zeros to ensure no division by zero errors
        tp = np.array([[0, 0, 0], [0, 0, 0]])
        fp = np.array([[0, 0, 0], [0, 0, 0]])
        fn = np.array([[0, 0, 0], [0, 0, 0]])
        tn = np.array([[0, 0, 0], [0, 0, 0]])

        expected_acc = np.array([0.0, 0.0])
        result = accuracy_micro(tp=tp, tn=tn, fn=fn, fp=fp, check_inputs=True)
        np.testing.assert_almost_equal(result, expected_acc)


class TestAccuracyMicro(unittest.TestCase):
    def test_accuracy_micro(self):
        tp = np.array([[10, 5, 0], [7, 8, 9]])
        tn = np.array([[10, 5, 0], [7, 8, 9]])
        fp = np.array([[2, 1, 0], [1, 0, 1]])
        fn = np.array([[1, 0, 2], [0, 2, 1]])

        tp_sum = np.sum(tp, axis=-1)
        tn_sum = np.sum(tn, axis=-1)
        fn_sum = np.sum(fn, axis=-1)
        fp_sum = np.sum(fp, axis=-1)

        expected_acc_micro = (tp_sum + tn_sum) / (
            tp_sum + tn_sum + fp_sum + fn_sum
        )

        result = accuracy_micro(tp=tp, tn=tn, fn=fn, fp=fp, check_inputs=True)

        np.testing.assert_almost_equal(result, expected_acc_micro)

    def test_zero_division_micro(self):
        # Test case with zeros to ensure no division by zero errors
        tp = np.array([[0, 0, 0], [0, 0, 0]])
        fp = np.array([[0, 0, 0], [0, 0, 0]])
        fn = np.array([[0, 0, 0], [0, 0, 0]])
        tn = np.array([[0, 0, 0], [0, 0, 0]])

        expected_acc = np.array([0.0, 0.0])
        result = accuracy_micro(tp=tp, tn=tn, fn=fn, fp=fp, check_inputs=True)
        np.testing.assert_almost_equal(result, expected_acc)
