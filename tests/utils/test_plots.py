import unittest
from typing import Tuple, List
import matplotlib.pyplot as plt
import math

# from matplotlib.testing.decorators import cleanup
import numpy as np

from streamauc.plot_util import create_square_subplots, plot_curve_and_auc


class TestCreateSquareSubplots(unittest.TestCase):

    def test_correct_number_of_subplots(self):
        num_subplots = 7
        fig, axs = create_square_subplots(num_subplots)
        self.assertEqual(
            len(axs),
            num_subplots,
            "The number of subplots created"
            " does not match the requested number.",
        )

    def test_subplot_layout(self):
        num_subplots = 7
        fig, axs = create_square_subplots(num_subplots)
        num_cols = math.ceil(math.sqrt(num_subplots))
        num_rows = math.ceil(num_subplots / num_cols)

        self.assertEqual(
            len(axs),
            num_subplots,
            "The number of axes in the figure "
            "does not match the requested number.",
        )

    def test_unused_subplots_hidden(self):
        num_subplots = 5
        fig, axs = create_square_subplots(num_subplots)

        num_cols = math.ceil(math.sqrt(num_subplots))
        num_rows = math.ceil(num_subplots / num_cols)

        total_subplots = num_rows * num_cols
        for i in range(num_subplots, total_subplots):
            with self.assertRaises(
                IndexError, msg=f"Axes index {i} should not exist."
            ):
                axs[i]

        visible_axes = axs
        self.assertEqual(
            len(visible_axes),
            num_subplots,
            "There are unused subplots that are not hidden.",
        )

    def test_single_subplot(self):
        num_subplots = 1
        fig, axs = create_square_subplots(num_subplots)
        self.assertEqual(
            len(axs),
            num_subplots,
            "The number of subplots created does not match the requested number.",
        )
        self.assertIsInstance(
            axs[0],
            plt.Axes,
            "The single subplot is not an instance of plt.Axes.",
        )

    def test_empty_case(self):
        num_subplots = 0
        with self.assertRaises(
            ValueError, msg="Creating 0 subplots should raise a ValueError."
        ):
            create_square_subplots(num_subplots)


class TestPlotcurveAUC(unittest.TestCase):

    def test_single_class_plot(self):
        fpr = np.array([0.0, 0.1, 0.2, 0.3, 1.0])
        tpr = np.array([0.0, 0.4, 0.6, 0.8, 1.0])
        thresholds = np.array([1.0, 0.9, 0.8, 0.7, 0.0])

        fig = plot_curve_and_auc(fpr, tpr, thresholds)
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)

    def test_multi_class_plot(self):
        fpr = np.array(
            [[0.0, 0.1, 0.2, 0.3, 1.0], [0.0, 0.2, 0.4, 0.6, 1.0]]
        ).T
        tpr = np.array(
            [[0.0, 0.4, 0.6, 0.8, 1.0], [0.0, 0.5, 0.7, 0.85, 1.0]]
        ).T
        thresholds = np.array([1.0, 0.9, 0.8, 0.7, 0.0])
        class_names = ["Class 1", "Class 2"]

        fig = plot_curve_and_auc(fpr, tpr, thresholds, class_names=class_names)
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)

    def test_invalid_fpr_shape(self):
        fpr = np.array([[[0.0, 0.1, 0.2, 0.3, 1.0]]])
        tpr = np.array([[[0.0, 0.4, 0.6, 0.8, 1.0]]])
        thresholds = np.array([1.0, 0.9, 0.8, 0.7, 0.0])

        with self.assertRaises(NotImplementedError):
            plot_curve_and_auc(fpr, tpr, thresholds)


if __name__ == "__main__":
    unittest.main()
