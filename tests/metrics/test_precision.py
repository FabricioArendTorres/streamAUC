import unittest
from unittest.mock import patch
import numpy as np

from streamauc.utils import *
from streamauc.metrics._precision import (
    precision_micro,
    precision_macro,
    precision_onevsall,
    precision,
)


class TestPrecisionFunction(unittest.TestCase):
    @patch("streamauc.metrics._precision.precision_micro")
    @patch("streamauc.metrics._precision.precision_macro")
    @patch("streamauc.metrics._precision.precision_onevsall")
    def test_precision_micro(self, mock_onevsall, mock_macro, mock_micro):
        tp = np.array([[10, 5, 0], [7, 8, 9]])
        fp = np.array([[2, 1, 0], [1, 0, 1]])

        mock_micro.return_value = np.array([0.83333333, 0.92307692])

        result = precision(
            tp, fp, method=AggregationMethod.MICRO, check_inputs=True
        )

        mock_micro.assert_called_once_with(tp=tp, fp=fp, check_inputs=True)
        mock_macro.assert_not_called()
        mock_onevsall.assert_not_called()

        np.testing.assert_almost_equal(result, mock_micro.return_value)

    @patch("streamauc.metrics._precision.precision_micro")
    @patch("streamauc.metrics._precision.precision_macro")
    @patch("streamauc.metrics._precision.precision_onevsall")
    def test_precision_macro(self, mock_onevsall, mock_macro, mock_micro):
        tp = np.array([[10, 5, 0], [7, 8, 9]])
        fp = np.array([[2, 1, 0], [1, 0, 1]])

        mock_macro.return_value = np.array([0.83333333, 0.88888889])

        result = precision(
            tp, fp, method=AggregationMethod.MACRO, check_inputs=True
        )

        mock_macro.assert_called_once_with(tp=tp, fp=fp, check_inputs=True)
        mock_micro.assert_not_called()
        mock_onevsall.assert_not_called()

        np.testing.assert_almost_equal(result, mock_macro.return_value)

    @patch("streamauc.metrics._precision.precision_micro")
    @patch("streamauc.metrics._precision.precision_macro")
    @patch("streamauc.metrics._precision.precision_onevsall")
    def test_precision_onevsall(self, mock_onevsall, mock_macro, mock_micro):
        tp = np.array([[10, 5, 0], [7, 8, 9]])
        fp = np.array([[2, 1, 0], [1, 0, 1]])
        class_index = 1

        mock_onevsall.return_value = np.array(
            [[0.83333333, 0.92307692], [0.8, 0.9]]
        )

        result = precision(
            tp,
            fp,
            method=AggregationMethod.ONE_VS_ALL,
            class_index=class_index,
            check_inputs=True,
        )

        mock_onevsall.assert_called_once_with(tp=tp, fp=fp, check_inputs=True)
        mock_micro.assert_not_called()
        mock_macro.assert_not_called()

        np.testing.assert_almost_equal(
            result, mock_onevsall.return_value[..., class_index]
        )

    def test_invalid_method(self):
        tp = np.array([[10, 5, 0], [7, 8, 9]])
        fp = np.array([[2, 1, 0], [1, 0, 1]])

        with self.assertRaises(ValueError):
            precision(tp, fp, method="invalid_method", check_inputs=True)


class TestComputePrecisionMicro(unittest.TestCase):
    def test_compute_precision_unit(self):
        # Test case with a simple example
        tp = np.array([[10, 5, 0], [7, 8, 9]])
        fp = np.array([[2, 1, 0], [1, 0, 1]])

        expected_precision = np.array([0.83333333, 0.92307692])

        precision = precision_micro(tp=tp, fp=fp, check_inputs=True)

        np.testing.assert_almost_equal(precision, expected_precision)

    def test_zero_division(self):
        # Test case with zeros to ensure no division by zero errors
        tp = np.array([[0, 0, 0], [0, 0, 0]])
        fp = np.array([[0, 0, 0], [0, 0, 0]])

        expected_precision = np.array([0.0, 0.0])

        precision = precision_micro(tp=tp, fp=fp, check_inputs=True)

        np.testing.assert_almost_equal(precision, expected_precision)

    def test_invalid_inputs(self):
        # Test case with invalid inputs to ensure proper error handling
        tp = np.array([[0, 0, 0], [0, -1, 0]])
        fp = np.array([[0, 0, 0], [0, 0, 0]])

        with self.assertRaises(AssertionError):
            precision = precision_micro(tp=tp, fp=fp, check_inputs=True)

        tp = np.array([[0, 0, 0, 0], [0, 1, 0, 0]])
        with self.assertRaises(AssertionError):
            precision = precision_micro(tp=tp, fp=fp, check_inputs=True)

        tp = np.array([[[0, 0, 0], [0, 1, 2]]])
        with self.assertRaises(AssertionError):
            precision = precision_micro(tp=tp, fp=fp, check_inputs=True)

        # This should not throw an error anymore
        precision = precision_micro(tp=tp, fp=fp, check_inputs=False)


class TestComputePrecisionMacro(unittest.TestCase):
    def test_compute_precision_unit(self):
        # Test case with a simple example
        tp = np.array([[10, 5, 0], [7, 8, 9]])
        fp = np.array([[2, 1, 0], [1, 0, 1]])

        expected_precision_indiv = np.array(
            [[0.83333333, 0.83333333, 0], [0.875, 1.0, 0.9]]
        )
        expected_precision = expected_precision_indiv.mean(-1, keepdims=False)

        precision = precision_macro(tp=tp, fp=fp, check_inputs=True)

        np.testing.assert_almost_equal(precision, expected_precision)

    def test_zero_division(self):
        # Test case with zeros to ensure no division by zero errors
        tp = np.array([[0, 0, 0], [0, 0, 0]])
        fp = np.array([[0, 0, 0], [0, 0, 0]])

        expected_precision = np.array([0.0, 0.0])

        precision = precision_macro(tp=tp, fp=fp, check_inputs=True)
        np.testing.assert_almost_equal(precision, expected_precision)

    def test_invalid_inputs(self):
        # Test case with invalid inputs to ensure proper error handling
        tp = np.array([[0, 0, 0], [0, -1, 0]])
        fp = np.array([[0, 0, 0], [0, 0, 0]])

        with self.assertRaises(AssertionError):
            precision = precision_macro(tp=tp, fp=fp, check_inputs=True)

        tp = np.array([[0, 0, 0, 0], [0, 1, 0, 0]])
        with self.assertRaises(AssertionError):
            precision = precision_macro(tp=tp, fp=fp, check_inputs=True)

        tp = np.array([[[0, 0, 0], [0, 1, 2]]])
        with self.assertRaises(AssertionError):
            precision = precision_macro(tp=tp, fp=fp, check_inputs=True)

        # This should not throw an error anymore
        precision = precision_macro(tp=tp, fp=fp, check_inputs=False)


class TestComputePrecisionOneVsAll(unittest.TestCase):
    def test_compute_precision_unit(self):
        tp = np.array([[10, 5, 0], [7, 8, 9]])
        fp = np.array([[2, 1, 0], [1, 0, 1]])

        expected_precision = np.array(
            [[0.83333333, 0.83333333, 0], [0.875, 1.0, 0.9]]
        )

        for i in range(tp.shape[-1]):
            precision = precision_onevsall(tp=tp, fp=fp, check_inputs=True)[
                :, i
            ]
            np.testing.assert_almost_equal(precision, expected_precision[:, i])

    def test_zero_division(self):
        # Test case with zeros to ensure no division by zero errors
        tp = np.array([[0, 0, 0], [0, 0, 0]])
        fp = np.array([[0, 0, 0], [0, 0, 0]])

        expected_precision = np.array([0.0, 0.0])

        for i in range(tp.shape[-1]):
            precision = precision_onevsall(tp=tp, fp=fp, check_inputs=True)[
                :, i
            ]
            np.testing.assert_almost_equal(precision, expected_precision)

    def test_invalid_inputs(self):
        # Test case with invalid inputs to ensure proper error handling
        tp = np.array([[0, 0, 0], [0, -1, 0]])
        fp = np.array([[0, 0, 0], [0, 0, 0]])

        with self.assertRaises(AssertionError):
            precision = precision_onevsall(tp=tp, fp=fp, check_inputs=True)

        tp = np.array([[0, 0, 0, 0], [0, 1, 0, 0]])
        with self.assertRaises(AssertionError):
            precision = precision_onevsall(tp=tp, fp=fp, check_inputs=True)

        tp = np.array([[[0, 0, 0], [0, 1, 2]]])
        with self.assertRaises(AssertionError):
            precision = precision_onevsall(tp=tp, fp=fp, check_inputs=True)

        # This should not throw an error anymore
        precision = precision_onevsall(tp=tp, fp=fp, check_inputs=False)


if __name__ == "__main__":
    unittest.main()
