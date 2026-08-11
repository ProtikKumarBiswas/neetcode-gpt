import numpy as np
from numpy.typing import NDArray
import math

class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: true labels (0 or 1)
        # y_pred: predicted probabilities
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        return round(-sum([((yi*math.log(pi))+((1-yi)*math.log(1-pi))) for yi, pi in zip(y_true, y_pred)])/len(y_pred), 4)

    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: one-hot encoded true labels (shape: n_samples x n_classes)
        # y_pred: predicted probabilities (shape: n_samples x n_classes)
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        s_arr = []
        for i in range(len(y_true[0])):
            for j in range(len(y_true)):
                s_arr.append(y_true[j][i]*math.log(y_pred[j][i]))
        return round(-sum(s_arr)/len(y_true), 4)
