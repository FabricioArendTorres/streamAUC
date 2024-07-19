import unittest
from unittest.mock import patch
import numpy as np

from streamauc.metrics._f1_score import (
    f1_score,
    f1_onevsall,
    f1_macro,
    f1_micro,
)
from streamauc.metrics._precision import precision_micro, precision_onevsall
from streamauc.metrics._tpr import tpr_micro, tpr_onevsall

from streamauc.utils import AggregationMethod


class TestF1ScoreFunction(unittest.TestCase):
    @patch("streamauc.metrics._f1_score.f1_micro")
    def test_f1_micro(self, mock_f1_micro):
        tp = np.array([[10, 5, 0], [7, 8, 9]])
        fp = np.array([[2, 1, 0], [1, 0, 1]])
        fn = np.array([[1, 0, 2], [0, 2, 1]])

        mock_f1_micro.return_value = np.array([0.83333333, 0.92307692])

        result = f1_score(
            tp, fp, fn, method=AggregationMethod.MICRO, check_inputs=True
        )

        mock_f1_micro.assert_called_once_with(
            tp=tp, fn=fn, fp=fp, check_inputs=True
        )

        np.testing.assert_almost_equal(result, mock_f1_micro.return_value)

    @patch("streamauc.metrics._f1_score.f1_macro")
    def test_f1_macro(self, mock_f1_macro):
        tp = np.array([[10, 5, 0], [7, 8, 9]])
        fp = np.array([[2, 1, 0], [1, 0, 1]])
        fn = np.array([[1, 0, 2], [0, 2, 1]])

        mock_f1_macro.return_value = np.array([0.83333333, 0.88888889])

        result = f1_score(
            tp, fp, fn, method=AggregationMethod.MACRO, check_inputs=True
        )

        mock_f1_macro.assert_called_once_with(
            tp=tp, fn=fn, fp=fp, check_inputs=True
        )

        np.testing.assert_almost_equal(result, mock_f1_macro.return_value)

    @patch("streamauc.metrics._f1_score.f1_onevsall")
    def test_f1_onevsall(self, mock_f1_onevsall):
        tp = np.array([[10, 5, 0], [7, 8, 9]])
        fp = np.array([[2, 1, 0], [1, 0, 1]])
        fn = np.array([[1, 0, 2], [0, 2, 1]])
        class_index = 1

        mock_f1_onevsall.return_value = np.array(
            [[0.83333333, 0.92307692], [0.8, 0.9]]
        )

        result = f1_score(
            tp,
            fp,
            fn,
            method=AggregationMethod.ONE_VS_ALL,
            class_index=class_index,
            check_inputs=True,
        )

        mock_f1_onevsall.assert_called_once_with(
            tp=tp, fn=fn, fp=fp, check_inputs=True
        )

        expected_f1 = mock_f1_onevsall.return_value[..., class_index]
        np.testing.assert_almost_equal(result, expected_f1)

    def test_invalid_method(self):
        tp = np.array([[10, 5, 0], [7, 8, 9]])
        fp = np.array([[2, 1, 0], [1, 0, 1]])
        fn = np.array([[1, 0, 2], [0, 2, 1]])

        with self.assertRaises(ValueError):
            f1_score(tp, fp, fn, method="invalid_method", check_inputs=True)


class TestF1OneVsAll(unittest.TestCase):
    def test_f1_onevsall(self):
        tp = np.array([[10, 5, 0], [7, 8, 9]])
        fp = np.array([[2, 1, 0], [1, 0, 1]])
        fn = np.array([[1, 0, 2], [0, 2, 1]])

        _precision = precision_onevsall(tp=tp, fp=fp, check_inputs=True)
        _tpr = tpr_onevsall(tp=tp, fn=fn, check_inputs=True)
        _expected_f1 = (2 * _precision * _tpr) / (_precision + _tpr + 1e-12)

        result = f1_onevsall(tp=tp, fp=fp, fn=fn, check_inputs=True)
        np.testing.assert_almost_equal(result, _expected_f1)

    def test_zero_division_onevsall(self):
        # Test case with zeros to ensure no division by zero errors
        tp = np.array([[0, 0, 0], [0, 0, 0]])
        fp = np.array([[0, 0, 0], [0, 0, 0]])
        fn = np.array([[0, 0, 0], [0, 0, 0]])

        _expected_f1 = np.array([0.0, 0.0])
        result = f1_onevsall(tp=tp, fp=fp, fn=fn, check_inputs=True)

        self.assertEqual(result.shape, tp.shape)

        np.testing.assert_almost_equal(result[:, 0], _expected_f1)
        np.testing.assert_almost_equal(result[:, 1], _expected_f1)
        np.testing.assert_almost_equal(result[:, 2], _expected_f1)


class TestF1Macro(unittest.TestCase):
    def test_f1_macro(self):
        tp = np.array([[10, 5, 0], [7, 8, 9]])
        fp = np.array([[2, 1, 0], [1, 0, 1]])
        fn = np.array([[1, 0, 2], [0, 2, 1]])

        _precision = precision_onevsall(tp=tp, fp=fp, check_inputs=True)
        _tpr = tpr_onevsall(tp=tp, fn=fn, check_inputs=True)
        _expected_f1_macro = (
            (2 * _precision * _tpr) / (_precision + _tpr + 1e-12)
        ).mean(-1)

        result = f1_macro(tp=tp, fp=fp, fn=fn, check_inputs=True)

        np.testing.assert_almost_equal(result, _expected_f1_macro)

    def test_zero_division_macro(self):
        # Test case with zeros to ensure no division by zero errors
        tp = np.array([[0, 0, 0], [0, 0, 0]])
        fp = np.array([[0, 0, 0], [0, 0, 0]])
        fn = np.array([[0, 0, 0], [0, 0, 0]])

        _expected_f1 = np.array([0.0, 0.0])
        result = f1_macro(tp=tp, fp=fp, fn=fn, check_inputs=True)
        np.testing.assert_almost_equal(result, _expected_f1)


#
class TestF1Micro(unittest.TestCase):
    def test_f1_micro(self):
        tp = np.array([[10, 5, 0], [7, 8, 9]])
        fp = np.array([[2, 1, 0], [1, 0, 1]])
        fn = np.array([[1, 0, 2], [0, 2, 1]])

        _precision = precision_micro(tp=tp, fp=fp, check_inputs=True)
        _tpr = tpr_micro(tp=tp, fn=fn, check_inputs=True)
        _expected_f1_micro = (2 * _precision * _tpr) / (
            _precision + _tpr + 1e-12
        )

        result = f1_micro(tp=tp, fp=fp, fn=fn, check_inputs=True)

        np.testing.assert_almost_equal(result, _expected_f1_micro)

    def test_zero_division_micro(self):
        # Test case with zeros to ensure no division by zero errors
        tp = np.array([[0, 0, 0], [0, 0, 0]])
        fp = np.array([[0, 0, 0], [0, 0, 0]])
        fn = np.array([[0, 0, 0], [0, 0, 0]])

        _expected_f1 = np.array([0.0, 0.0])
        result = f1_micro(tp=tp, fp=fp, fn=fn, check_inputs=True)
        np.testing.assert_almost_equal(result, _expected_f1)
