import numpy as np
from numpy.typing import NDArray
import math


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        z_std = z - max(z)
        e_z_arr = [math.exp(zj) for zj in z_std]
        sum_e_z = sum(e_z_arr)
        return [round(e_z_j/sum_e_z, 4) for e_z_j in e_z_arr]