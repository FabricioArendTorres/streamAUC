import math
from typing import List, Optional, Tuple

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection

from streamauc.utils import auc


def create_square_subplots(
    num_subplots,
) -> Tuple[plt.Figure, List[plt.Axes]]:  # pragma: nocover

    if num_subplots <= 0:
        raise ValueError("Must be greater 0...")

    # Calculate the number of rows and columns
    num_cols = math.ceil(math.sqrt(num_subplots))
    num_rows = math.ceil(num_subplots / num_cols)

    # Create the subplots
    fig, axs = plt.subplots(
        num_rows, num_cols, figsize=(num_cols * 5, num_rows * 5)
    )

    # Flatten the axs array for easy iteration,
    # even if there is only one row/column
    if num_subplots == 1:
        axs = [axs]
    else:
        axs = axs.flatten()

    # Hide any unused subplots
    for i in range(num_subplots, num_rows * num_cols):
        fig.delaxes(axs[i])

    return fig, fig.axes


def plot_curve_and_auc(
    x_values: np.ndarray,
    y_values: np.ndarray,
    thresholds: np.ndarray,
    class_names: Optional[List[str]] = None,
    cmap="viridis",
) -> plt.Figure:
    if x_values.ndim == 1:

        fig, axs = create_square_subplots(1)
        ax = axs[0]

        # Normalization for color mapping
        norm = plt.Normalize(vmin=min(thresholds), vmax=max(thresholds))
        cmap = plt.get_cmap(cmap)

        _auc = auc(x_values, y_values)
        points = np.array([x_values, y_values]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap=cmap, norm=norm)
        lc.set_array(thresholds)
        lc.set_linewidth(2)
        ax.add_collection(lc)
        ax.scatter(x_values, y_values, color="black", marker="+", alpha=0.1)

        if class_names is None:
            ax.set_title(f"AUC={_auc:.3f}")
        else:
            ax.set_title(f"{class_names[0]} | AUC={_auc:.3f}")

        ax.set_ylabel("TPR")
        ax.set_xlabel("FPR")
        return fig

    elif x_values.ndim == 2:
        num_classes = y_values.shape[-1]

        if class_names is not None and len(class_names) != num_classes:
            raise ValueError(
                "Class names must be of same length as the "
                "fpr.shape[-1] and tpr.shape[-1]."
            )
        elif class_names is None:
            class_names = [f"Class_{i}" for i in range(num_classes)]

        fig, axs = create_square_subplots(num_classes)
        # Normalization for color mapping
        norm = plt.Normalize(vmin=min(thresholds), vmax=max(thresholds))
        cmap = plt.get_cmap("viridis")

        for ax, i in zip(axs, range(num_classes)):
            _auc = auc(x_values[:, i], y_values[:, i])
            points = np.array([x_values[:, i], y_values[:, i]]).T.reshape(
                -1, 1, 2
            )
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            lc = LineCollection(segments, cmap=cmap, norm=norm)
            lc.set_array(thresholds)
            lc.set_linewidth(2)
            ax.add_collection(lc)
            ax.scatter(
                x_values[:, i],
                y_values[:, i],
                color="black",
                marker="+",
                alpha=0.1,
            )
            ax.set_title(f"{class_names[i]} | AUC={_auc:.3f}")

            ax.set_ylabel("TPR")
            ax.set_xlabel("FPR")

        # Create a ScalarMappable and add a color bar
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])  # Only needed for the color bar

        # Add color bar to the figure
        fig.colorbar(
            sm,
            ax=axs,
            orientation="horizontal",
            fraction=0.02,
            pad=0.1,
            aspect=30,
            label="Threshold",
        )
        fig.subplots_adjust(hspace=0.4, wspace=0.4, bottom=0.2)
        return fig
    else:
        raise NotImplementedError("...")
