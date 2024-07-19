import unittest
from unittest.mock import patch
import numpy as np

from streamauc.utils import *
from streamauc.metrics._fpr import fpr_micro, fpr_macro, fpr_onevsall, fpr


# Test case with a simple example
tp = np.array([[10, 5, 0], [7, 8, 9]])
fp = np.array([[2, 1, 0], [1, 0, 1]])
fn = np.array([[1, 0, 2], [0, 2, 1]])
tn = np.array([[5, 10, 15], [20, 15, 10]])


class TestFPRFunction(unittest.TestCase):
    @patch("streamauc.metrics._fpr.fpr_micro")
    @patch("streamauc.metrics._fpr.fpr_macro")
    @patch("streamauc.metrics._fpr.fpr_onevsall")
    def test_fpr_micro(self, mock_onevsall, mock_macro, mock_micro):
        fp = np.array([[2, 1, 0], [1, 0, 1]])
        tn = np.array([[10, 5, 0], [7, 8, 9]])

        mock_micro.return_value = np.array([0.16666667, 0.07692308])

        result = fpr(fp, tn, method=AggregationMethod.MICRO, check_inputs=True)

        mock_micro.assert_called_once_with(fp=fp, tn=tn, check_inputs=True)
        mock_macro.assert_not_called()
        mock_onevsall.assert_not_called()

        np.testing.assert_almost_equal(result, mock_micro.return_value)

    @patch("streamauc.metrics._fpr.fpr_micro")
    @patch("streamauc.metrics._fpr.fpr_macro")
    @patch("streamauc.metrics._fpr.fpr_onevsall")
    def test_fpr_macro(self, mock_onevsall, mock_macro, mock_micro):
        fp = np.array([[2, 1, 0], [1, 0, 1]])
        tn = np.array([[10, 5, 0], [7, 8, 9]])

        mock_macro.return_value = np.array([0.16666667, 0.11111111])

        result = fpr(fp, tn, method=AggregationMethod.MACRO, check_inputs=True)

        mock_macro.assert_called_once_with(fp=fp, tn=tn, check_inputs=True)
        mock_micro.assert_not_called()
        mock_onevsall.assert_not_called()

        np.testing.assert_almost_equal(result, mock_macro.return_value)

    @patch("streamauc.metrics._fpr.fpr_micro")
    @patch("streamauc.metrics._fpr.fpr_macro")
    @patch("streamauc.metrics._fpr.fpr_onevsall")
    def test_fpr_onevsall(self, mock_onevsall, mock_macro, mock_micro):
        fp = np.array([[2, 1, 0], [1, 0, 1]])
        tn = np.array([[10, 5, 0], [7, 8, 9]])
        class_index = 1

        mock_onevsall.return_value = np.array(
            [[0.16666667, 0.07692308], [0.125, 0.1]]
        )

        result = fpr(
            fp,
            tn,
            method=AggregationMethod.ONE_VS_ALL,
            class_index=class_index,
            check_inputs=True,
        )

        mock_onevsall.assert_called_once_with(fp=fp, tn=tn, check_inputs=True)
        mock_micro.assert_not_called()
        mock_macro.assert_not_called()

        np.testing.assert_almost_equal(
            result, mock_onevsall.return_value[..., class_index]
        )

    def test_invalid_method(self):
        fp = np.array([[2, 1, 0], [1, 0, 1]])
        tn = np.array([[10, 5, 0], [7, 8, 9]])

        with self.assertRaises(ValueError):
            fpr(fp, tn, method="invalid_method", check_inputs=True)


class TestComputeFprMicro(unittest.TestCase):
    def test_compute_roc_micro(self):
        expected_fpr = np.array([0.09090909, 0.04255319])

        fpr = fpr_micro(fp=fp, tn=tn)

        np.testing.assert_almost_equal(fpr, expected_fpr)

    def test_zero_division(self):
        # Test case with zeros to ensure no division by zero errors
        fp = np.array([[0, 0, 0], [0, 0, 0]])
        tn = np.array([[0, 0, 0], [0, 0, 0]])

        expected_fpr = np.array([0.0, 0.0])

        fpr = fpr_micro(fp=fp, tn=tn)

        np.testing.assert_almost_equal(fpr, expected_fpr)

    def test_invalid_inputs(self):
        # Test case with zeros to ensure no division by zero errors
        fp = np.array([[0, 0, 0], [0, -1, 0]])
        tn = np.array([[0, 0, 0], [0, 0, 0]])

        with self.assertRaises(AssertionError):
            fpr = fpr_micro(fp=fp, tn=tn)

        fp = np.array([[0, 0, 0, 0], [0, 1, 0, 0]])
        with self.assertRaises(AssertionError):
            fpr = fpr_micro(fp=fp, tn=tn)

        fp = np.array([[[0, 0, 0], [0, 1, 2]]])
        with self.assertRaises(AssertionError):
            fpr = fpr_micro(fp=fp, tn=tn)

        fpr = fpr_micro(fp=fp, tn=tn, check_inputs=False)


class TestComputeFprMacro(unittest.TestCase):

    def test_compute_roc_macro(self):
        # Test case with a simple example
        fp = np.array([[2, 1], [1, 0]])
        tn = np.array([[5, 10], [20, 15]])

        tpr_classwise = np.array([[0.90909091, 1.0], [1.0, 0.8]])

        fpr_classwise = np.array([[0.28571429, 0.09090909], [0.04761905, 0.0]])

        expected_tpr = np.mean(tpr_classwise, -1)
        expected_fpr = np.mean(fpr_classwise, -1)

        fpr = fpr_macro(fp=fp, tn=tn)

        np.testing.assert_almost_equal(fpr, expected_fpr)

    def test_zero_division(self):
        # Test case with zeros to ensure no division by zero errors
        fp = np.array([[0, 0], [0, 0]])
        tn = np.array([[0, 0], [0, 0]])

        expected_fpr = np.array([0.0, 0.0])

        fpr = fpr_macro(fp=fp, tn=tn)

        np.testing.assert_almost_equal(fpr, expected_fpr)

    def test_invalid_inputs(self):
        # Test case with zeros to ensure no division by zero errors
        fp = np.array([[0, 0, 0], [0, -1, 0]])
        tn = np.array([[0, 0, 0], [0, 0, 0]])

        with self.assertRaises(AssertionError):
            fpr = fpr_macro(fp=fp, tn=tn)

        tp = np.array([[0, 0, 0, 0], [0, 1, 0, 0]])
        with self.assertRaises(AssertionError):
            fpr = fpr_macro(fp=fp, tn=tn)

        tp = np.array([[[0, 0, 0], [0, 1, 2]]])
        with self.assertRaises(AssertionError):
            fpr = fpr_macro(fp=fp, tn=tn)

        fpr = fpr_macro(fp=fp, tn=tn, check_inputs=False)


class TestComputeFpr1vsAll(unittest.TestCase):

    def test_compute_roc_1vsall(self):
        # Test case with a simple example
        fp = np.array([[2, 1], [1, 0]])
        tn = np.array([[5, 10], [20, 15]])

        expected_fpr = np.array([[0.28571429, 0.09090909], [0.04761905, 0.0]])

        for i in range(tn.shape[1]):
            fpr = fpr_onevsall(fp=fp, tn=tn)[:, i]
            np.testing.assert_almost_equal(fpr, expected_fpr[:, i])

    def test_zero_division_1vsall(self):
        # Test case with zeros to ensure no division by zero errors
        fp = np.array([[0, 0], [0, 0]])
        tn = np.array([[0, 0], [0, 0]])
        class_index = 0

        expected_fpr = np.array([0.0, 0.0])

        fpr = fpr_onevsall(fp=fp, tn=tn)[:, class_index]

        np.testing.assert_almost_equal(fpr, expected_fpr)

    def test_invalid_inputs_1vsall(self):
        # Test case with zeros to ensure no division by zero errors
        fp = np.array([[0, 0, 0], [0, -1, 0]])
        tn = np.array([[0, 0, 0], [0, 0, 0]])

        with self.assertRaises(AssertionError):
            fpr = fpr_onevsall(fp=fp, tn=tn)

        fp = np.array([[0, 0, 0, 0], [0, 1, 0, 0]])
        with self.assertRaises(AssertionError):
            fpr = fpr_onevsall(fp=fp, tn=tn)

        fp = np.array([[[0, 0, 0], [0, 1, 2]]])
        with self.assertRaises(AssertionError):
            fpr = fpr_onevsall(fp=fp, tn=tn)


if __name__ == "__main__":
    unittest.main()
