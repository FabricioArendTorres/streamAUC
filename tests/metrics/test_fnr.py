import unittest
from unittest.mock import patch
import numpy as np

from streamauc.metrics._fnr import fnr

from streamauc.utils import AggregationMethod


class TestFNRFunction(unittest.TestCase):
    @patch("streamauc.metrics._tpr.tpr_micro")
    def test_fnr_micro(self, mock_tpr):
        tp = np.array([[10, 5, 0], [7, 8, 9]])
        fn = np.array([[1, 0, 2], [0, 2, 1]])

        mock_tpr.return_value = np.array([0.83333333, 0.92307692])

        result = fnr(tp, fn, method=AggregationMethod.MICRO, check_inputs=True)

        mock_tpr.assert_called_once_with(tp=tp, fn=fn, check_inputs=True)

        expected_fnr = 1 - mock_tpr.return_value
        np.testing.assert_almost_equal(result, expected_fnr)

    @patch("streamauc.metrics._tpr.tpr_macro")
    def test_fnr_macro(self, mock_tpr):
        tp = np.array([[10, 5, 0], [7, 8, 9]])
        fn = np.array([[1, 0, 2], [0, 2, 1]])

        mock_tpr.return_value = np.array([0.83333333, 0.88888889])

        result = fnr(tp, fn, method=AggregationMethod.MACRO, check_inputs=True)

        mock_tpr.assert_called_once_with(tp=tp, fn=fn, check_inputs=True)

        expected_fnr = 1 - mock_tpr.return_value
        np.testing.assert_almost_equal(result, expected_fnr)

    @patch("streamauc.metrics._tpr.tpr_onevsall")
    def test_fnr_onevsall(self, mock_tpr):
        tp = np.array([[10, 5, 0], [7, 8, 9]])
        fn = np.array([[1, 0, 2], [0, 2, 1]])
        class_index = 1

        mock_tpr.return_value = np.array(
            [[0.83333333, 0.92307692], [0.8, 0.9]]
        )

        result = fnr(
            tp,
            fn,
            method=AggregationMethod.ONE_VS_ALL,
            class_index=class_index,
            check_inputs=True,
        )
        mock_tpr.assert_called_once_with(tp=tp, fn=fn, check_inputs=True)
        expected_fnr = 1 - mock_tpr.return_value[..., class_index]

        np.testing.assert_almost_equal(result, expected_fnr)

    def test_invalid_method(self):
        tp = np.array([[10, 5, 0], [7, 8, 9]])
        fn = np.array([[1, 0, 2], [0, 2, 1]])

        with self.assertRaises(ValueError):
            fnr(tp, fn, method="invalid_method", check_inputs=True)


if __name__ == "__main__":
    unittest.main()
