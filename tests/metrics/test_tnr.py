# test_tnr.py
import unittest
from unittest.mock import patch
import numpy as np

from streamauc.metrics._tnr import tnr
from streamauc.utils import AggregationMethod


class TestTNRFunction(unittest.TestCase):
    @patch("streamauc.metrics._fpr.fpr_micro")
    def test_tnr_micro(self, mock_fpr):
        fp = np.array([[2, 1, 0], [1, 0, 1]])
        tn = np.array([[10, 5, 0], [7, 8, 9]])

        mock_fpr.return_value = np.array([0.16666667, 0.07692308])

        result = tnr(fp, tn, method=AggregationMethod.MICRO, check_inputs=True)

        mock_fpr.assert_called_once_with(fp=fp, tn=tn, check_inputs=True)

        expected_tnr = 1 - mock_fpr.return_value
        np.testing.assert_almost_equal(result, expected_tnr)

    @patch("streamauc.metrics._fpr.fpr_macro")
    def test_tnr_macro(self, mock_fpr):
        fp = np.array([[2, 1, 0], [1, 0, 1]])
        tn = np.array([[10, 5, 0], [7, 8, 9]])

        mock_fpr.return_value = np.array([0.16666667, 0.11111111])

        result = tnr(fp, tn, method=AggregationMethod.MACRO, check_inputs=True)

        mock_fpr.assert_called_once_with(fp=fp, tn=tn, check_inputs=True)

        expected_tnr = 1 - mock_fpr.return_value
        np.testing.assert_almost_equal(result, expected_tnr)

    @patch("streamauc.metrics._fpr.fpr_onevsall")
    def test_tnr_onevsall(self, mock_fpr):
        fp = np.array([[2, 1, 0], [1, 0, 1]])
        tn = np.array([[10, 5, 0], [7, 8, 9]])
        class_index = 1

        mock_fpr.return_value = np.array(
            [[0.16666667, 0.07692308], [0.125, 0.1]]
        )

        result = tnr(
            fp,
            tn,
            method=AggregationMethod.ONE_VS_ALL,
            class_index=class_index,
            check_inputs=True,
        )

        mock_fpr.assert_called_once_with(fp=fp, tn=tn, check_inputs=True)

        expected_tnr = 1 - mock_fpr.return_value[..., class_index]
        np.testing.assert_almost_equal(result, expected_tnr)

    def test_invalid_method(self):
        fp = np.array([[2, 1, 0], [1, 0, 1]])
        tn = np.array([[10, 5, 0], [7, 8, 9]])

        with self.assertRaises(ValueError):
            tnr(fp, tn, method="invalid_method", check_inputs=True)


if __name__ == "__main__":
    unittest.main()
