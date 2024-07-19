import unittest
from unittest.mock import patch
import numpy as np

from streamauc.metrics._tpr import tpr_micro, tpr_macro, tpr_onevsall, tpr
from streamauc.utils import *


# Test case with a simple example
tp = np.array([[10, 5, 0], [7, 8, 9]])
fp = np.array([[2, 1, 0], [1, 0, 1]])
fn = np.array([[1, 0, 2], [0, 2, 1]])
tn = np.array([[5, 10, 15], [20, 15, 10]])


class TestTPRFunction(unittest.TestCase):
    @patch("streamauc.metrics._tpr.tpr_micro")
    @patch("streamauc.metrics._tpr.tpr_macro")
    @patch("streamauc.metrics._tpr.tpr_onevsall")
    def test_tpr_micro(self, mock_onevsall, mock_macro, mock_micro):
        tp = np.array([[10, 5, 0], [7, 8, 9]])
        fn = np.array([[1, 0, 2], [0, 2, 1]])

        mock_micro.return_value = np.array([0.83333333, 0.92307692])

        result = tpr(tp, fn, method=AggregationMethod.MICRO, check_inputs=True)

        mock_micro.assert_called_once_with(tp=tp, fn=fn, check_inputs=True)
        mock_macro.assert_not_called()
        mock_onevsall.assert_not_called()

        np.testing.assert_almost_equal(result, mock_micro.return_value)

    @patch("streamauc.metrics._tpr.tpr_micro")
    @patch("streamauc.metrics._tpr.tpr_macro")
    @patch("streamauc.metrics._tpr.tpr_onevsall")
    def test_tpr_macro(self, mock_onevsall, mock_macro, mock_micro):
        tp = np.array([[10, 5, 0], [7, 8, 9]])
        fn = np.array([[1, 0, 2], [0, 2, 1]])

        mock_macro.return_value = np.array([0.83333333, 0.88888889])

        result = tpr(tp, fn, method=AggregationMethod.MACRO, check_inputs=True)

        mock_macro.assert_called_once_with(tp=tp, fn=fn, check_inputs=True)
        mock_micro.assert_not_called()
        mock_onevsall.assert_not_called()

        np.testing.assert_almost_equal(result, mock_macro.return_value)

    @patch("streamauc.metrics._tpr.tpr_micro")
    @patch("streamauc.metrics._tpr.tpr_macro")
    @patch("streamauc.metrics._tpr.tpr_onevsall")
    def test_tpr_onevsall(self, mock_onevsall, mock_macro, mock_micro):
        tp = np.array([[10, 5, 0], [7, 8, 9]])
        fn = np.array([[1, 0, 2], [0, 2, 1]])
        class_index = 1

        mock_onevsall.return_value = np.array(
            [[0.83333333, 0.92307692], [0.8, 0.9]]
        )

        result = tpr(
            tp,
            fn,
            method=AggregationMethod.ONE_VS_ALL,
            class_index=class_index,
            check_inputs=True,
        )

        mock_onevsall.assert_called_once_with(tp=tp, fn=fn, check_inputs=True)
        mock_micro.assert_not_called()
        mock_macro.assert_not_called()

        np.testing.assert_almost_equal(
            result, mock_onevsall.return_value[..., class_index]
        )

    def test_invalid_method(self):
        tp = np.array([[10, 5, 0], [7, 8, 9]])
        fn = np.array([[1, 0, 2], [0, 2, 1]])

        with self.assertRaises(ValueError):
            tpr(tp, fn, method="invalid_method", check_inputs=True)


class TestComputeTprMicro(unittest.TestCase):
    def test_compute_tpr_micro(self):
        expected_tpr = np.array([0.83333333, 0.88888889])
        _tpr = tpr_micro(tp=tp, fn=fn)
        np.testing.assert_almost_equal(_tpr, expected_tpr)

    def test_zero_division(self):
        # Test case with zeros to ensure no division by zero errors
        tp = np.array([[0, 0, 0], [0, 0, 0]])
        fn = np.array([[0, 0, 0], [0, 0, 0]])

        expected_tpr = np.array([0.0, 0.0])

        _tpr = tpr_micro(tp=tp, fn=fn)

        np.testing.assert_almost_equal(_tpr, expected_tpr)

    def test_invalid_inputs(self):
        # Test case with zeros to ensure no division by zero errors
        tp = np.array([[0, 0, 0], [0, -1, 0]])
        fn = np.array([[0, 0, 0], [0, 0, 0]])

        with self.assertRaises(AssertionError):
            _tpr = tpr_micro(tp=tp, fn=fn)

        tp = np.array([[0, 0, 0, 0], [0, 1, 0, 0]])
        with self.assertRaises(AssertionError):
            _tpr = tpr_micro(tp=tp, fn=fn)

        tp = np.array([[[0, 0, 0], [0, 1, 2]]])
        with self.assertRaises(AssertionError):
            _tpr = tpr_micro(tp=tp, fn=fn)

        _tpr = tpr_micro(tp=tp, fn=fn, check_inputs=False)


class TestComputeTprMacro(unittest.TestCase):
    def test_compute_roc_macro(self):
        # Test case with a simple example
        tp = np.array([[10, 5], [7, 8]])
        fn = np.array([[1, 0], [0, 2]])

        tpr_classwise = np.array([[0.90909091, 1.0], [1.0, 0.8]])

        expected_tpr = np.mean(tpr_classwise, -1)

        _tpr = tpr_macro(tp=tp, fn=fn)
        np.testing.assert_almost_equal(_tpr, expected_tpr)

    def test_zero_division(self):
        # Test case with zeros to ensure no division by zero errors
        tp = np.array([[0, 0], [0, 0]])
        fn = np.array([[0, 0], [0, 0]])

        expected_tpr = np.array([0.0, 0.0])

        _tpr = tpr_macro(tp=tp, fn=fn)

        np.testing.assert_almost_equal(_tpr, expected_tpr)

    def test_invalid_inputs(self):
        # Test case with zeros to ensure no division by zero errors
        tp = np.array([[0, 0, 0], [0, 0, -1]])
        fp = np.array([[0, 0, 0], [0, 0, 0]])
        fn = np.array([[0, 0, 0], [0, 0, 0]])
        tn = np.array([[0, 0, 0], [0, 0, 0]])

        with self.assertRaises(AssertionError):
            _tpr = tpr_macro(tp=tp, fn=fn)

        tp = np.array([[0, 0, 0, 0], [0, 1, 0, 0]])
        with self.assertRaises(AssertionError):
            _tpr = tpr_macro(tp=tp, fn=fn)

        tp = np.array([[[0, 0, 0], [0, 1, 2]]])
        with self.assertRaises(AssertionError):
            _tpr = tpr_macro(tp=tp, fn=fn)

        _tpr = tpr_macro(tp=tp, fn=fn, check_inputs=False)


class TestComputeRoc1vsAll(unittest.TestCase):

    def test_compute_roc_1vsall(self):
        # Test case with a simple example
        tp = np.array([[10, 5], [7, 8]])
        fn = np.array([[1, 0], [0, 2]])

        expected_tpr = np.array([[0.90909091, 1.0], [1.0, 0.8]])

        for i in range(tp.shape[1]):
            _tpr = tpr_onevsall(tp=tp, fn=fn)[:, i]
            np.testing.assert_almost_equal(_tpr, expected_tpr[:, i])

    def test_zero_division_1vsall(self):
        # Test case with zeros to ensure no division by zero errors
        tp = np.array([[0, 0], [0, 0]])
        fn = np.array([[0, 0], [0, 0]])
        class_index = 0

        expected_tpr = np.array([0.0, 0.0])

        _tpr = tpr_onevsall(tp=tp, fn=fn)[:, class_index]

        np.testing.assert_almost_equal(_tpr, expected_tpr)

    def test_invalid_inputs_1vsall(self):
        # Test case with zeros to ensure no division by zero errors
        tp = np.array([[0, 0, 0], [0, 0, -1]])
        fn = np.array([[0, 0, 0], [0, 0, 0]])

        with self.assertRaises(AssertionError):
            _tpr = tpr_onevsall(tp=tp, fn=fn)

        tp = np.array([[0, 0, 0, 0], [0, 1, 0, 0]])
        with self.assertRaises(AssertionError):
            _tpr = tpr_macro(tp=tp, fn=fn)

        tp = np.array([[[0, 0, 0], [0, 1, 2]]])
        with self.assertRaises(AssertionError):
            _tpr = tpr_macro(tp=tp, fn=fn)


if __name__ == "__main__":
    unittest.main()
