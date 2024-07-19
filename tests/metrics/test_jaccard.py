import unittest
import numpy as np
from unittest.mock import patch
from streamauc.metrics._jaccard_index import (
    jaccard_index,
    jaccard_index_micro,
    jaccard_index_macro,
    jaccard_index_onevsall,
)
from streamauc.utils import AggregationMethod


class TestJaccardIndexFunction(unittest.TestCase):
    @patch("streamauc.metrics._jaccard_index.jaccard_index_micro")
    def test_jaccard_index_micro(self, mock_jaccard_micro):
        tp = np.array([[10, 5, 0], [7, 8, 9]])
        fp = np.array([[2, 1, 0], [1, 0, 1]])
        fn = np.array([[1, 0, 2], [0, 2, 1]])

        mock_jaccard_micro.return_value = np.array([0.76923077, 0.81818182])

        result = jaccard_index(
            tp, fp, fn, method=AggregationMethod.MICRO, check_inputs=True
        )

        mock_jaccard_micro.assert_called_once_with(
            tp=tp, fn=fn, fp=fp, check_inputs=True
        )

        np.testing.assert_almost_equal(result, mock_jaccard_micro.return_value)

    @patch("streamauc.metrics._jaccard_index.jaccard_index_macro")
    def test_jaccard_index_macro(self, mock_jaccard_macro):
        tp = np.array([[10, 5, 0], [7, 8, 9]])
        fp = np.array([[2, 1, 0], [1, 0, 1]])
        fn = np.array([[1, 0, 2], [0, 2, 1]])

        mock_jaccard_macro.return_value = np.array([0.66666667, 0.85714286])

        result = jaccard_index(
            tp, fp, fn, method=AggregationMethod.MACRO, check_inputs=True
        )

        mock_jaccard_macro.assert_called_once_with(
            tp=tp, fn=fn, fp=fp, check_inputs=True
        )

        np.testing.assert_almost_equal(result, mock_jaccard_macro.return_value)

    @patch("streamauc.metrics._jaccard_index.jaccard_index_onevsall")
    def test_jaccard_index_onevsall(self, mock_jaccard_onevsall):
        tp = np.array([[10, 5, 0], [7, 8, 9]])
        fp = np.array([[2, 1, 0], [1, 0, 1]])
        fn = np.array([[1, 0, 2], [0, 2, 1]])
        class_index = 1

        mock_jaccard_onevsall.return_value = np.array(
            [[0.76923077, 0.83333333, 0], [0.875, 1.0, 0.9]]
        )

        result = jaccard_index(
            tp,
            fp,
            fn,
            method=AggregationMethod.ONE_VS_ALL,
            class_index=class_index,
            check_inputs=True,
        )

        mock_jaccard_onevsall.assert_called_once_with(
            tp=tp, fn=fn, fp=fp, check_inputs=True
        )

        expected_jaccard = mock_jaccard_onevsall.return_value[..., class_index]
        np.testing.assert_almost_equal(result, expected_jaccard)

    def test_invalid_method(self):
        tp = np.array([[10, 5, 0], [7, 8, 9]])
        fp = np.array([[2, 1, 0], [1, 0, 1]])
        fn = np.array([[1, 0, 2], [0, 2, 1]])

        with self.assertRaises(ValueError):
            jaccard_index(
                tp, fp, fn, method="invalid_method", check_inputs=True
            )


class TestJaccardIndexOneVsAll(unittest.TestCase):
    def test_jaccard_index_onevsall(self):
        tp = np.array([[10, 5, 0], [7, 8, 9]])
        fp = np.array([[2, 1, 0], [1, 0, 1]])
        fn = np.array([[1, 0, 2], [0, 2, 1]])

        expected_jaccard = tp / (tp + fn + fp)
        result = jaccard_index_onevsall(tp=tp, fn=fn, fp=fp, check_inputs=True)

        np.testing.assert_almost_equal(result, expected_jaccard)

    def test_zero_division_onevsall(self):
        # Test case with zeros to ensure no division by zero errors
        tp = np.array([[0, 0, 0], [0, 0, 0]])
        fp = np.array([[0, 0, 0], [0, 0, 0]])
        fn = np.array([[0, 0, 0], [0, 0, 0]])

        expected_jaccard = np.array([0.0, 0.0])
        result = jaccard_index_onevsall(tp=tp, fp=fp, fn=fn, check_inputs=True)
        self.assertEqual(result.shape, tp.shape)
        np.testing.assert_almost_equal(result[:, 0], expected_jaccard)
        np.testing.assert_almost_equal(result[:, 1], expected_jaccard)


class TestJaccardIndexMacro(unittest.TestCase):
    def test_jaccard_index_macro(self):
        tp = np.array([[10, 5, 0], [7, 8, 9]])
        fp = np.array([[2, 1, 0], [1, 0, 1]])
        fn = np.array([[1, 0, 2], [0, 2, 1]])

        expected_jaccard_onevsall = tp / (tp + fn + fp)
        expected_jaccard_macro = np.mean(expected_jaccard_onevsall, axis=-1)

        result = jaccard_index_macro(tp=tp, fn=fn, fp=fp, check_inputs=True)

        np.testing.assert_almost_equal(result, expected_jaccard_macro)

    def test_zero_division_macro(self):
        # Test case with zeros to ensure no division by zero errors
        tp = np.array([[0, 0, 0], [0, 0, 0]])
        fp = np.array([[0, 0, 0], [0, 0, 0]])
        fn = np.array([[0, 0, 0], [0, 0, 0]])

        _expected_jaccard = np.array([0.0, 0.0])
        result = jaccard_index_macro(tp=tp, fp=fp, fn=fn, check_inputs=True)
        np.testing.assert_almost_equal(result, _expected_jaccard)


class TestJaccardIndexMicro(unittest.TestCase):
    def test_jaccard_index_micro(self):
        tp = np.array([[10, 5, 0], [7, 8, 9]])
        fp = np.array([[2, 1, 0], [1, 0, 1]])
        fn = np.array([[1, 0, 2], [0, 2, 1]])

        tp_sum = np.sum(tp, axis=-1)
        fn_sum = np.sum(fn, axis=-1)
        fp_sum = np.sum(fp, axis=-1)

        expected_jaccard = tp_sum / (tp_sum + fn_sum + fp_sum)
        result = jaccard_index_micro(tp=tp, fn=fn, fp=fp, check_inputs=True)

        np.testing.assert_almost_equal(result, expected_jaccard)

    def test_zero_division_micro(self):
        # Test case with zeros to ensure no division by zero errors
        tp = np.array([[0, 0, 0], [0, 0, 0]])
        fp = np.array([[0, 0, 0], [0, 0, 0]])
        fn = np.array([[0, 0, 0], [0, 0, 0]])

        _expected_jaccard = np.array([0.0, 0.0])
        result = jaccard_index_micro(tp=tp, fp=fp, fn=fn, check_inputs=True)
        np.testing.assert_almost_equal(result, _expected_jaccard)
