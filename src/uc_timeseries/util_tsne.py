import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import manifold


class AnomalyTSNE:
    """Use this class to visualize is anomaly label make sense."""

    def __init__(self, X_df: pd.DataFrame, y: np.ndarray):
        tsne = manifold.TSNE(n_components=2, init="pca", random_state=0)
        self.X = X_df
        self.X = tsne.fit_transform(X_df)
        self.y = y
        self.result_df = pd.DataFrame(self.X)
        self.result_df["preds"] = y

    def plot_embedding(self, title=None):
        """Plot tsne. X must be 2-dimention.

        Args:
            X (pd.DataFrame): Input features of 2-dim
            y ([type], optional): Label array. Defaults to None.
            title ([type], optional): Graph title. Defaults to None.
        """
        x_min, x_max = np.min(self.X, 0), np.max(self.X, 0)
        self.X = (self.X - x_min) / (x_max - x_min)
        plt.figure()
        plt.subplot(111)

        for i in range(self.X.shape[0]):
            plt.text(
                self.X[i, 0],
                self.X[i, 1],
                s="*" if self.y is None else self.y[i],  # Only show label when available
                color="red"
                if self.y is None
                else plt.cm.Set1(self.y[i]),  # Use different color for label values
                fontdict={"weight": "bold", "size": 9},
            )

        plt.xticks([]), plt.yticks([])
        if title is not None:
            plt.title(title)

        plt.show()
