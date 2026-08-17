import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        # Loss: MSE = mean((predictions - y_true)^2)
        #
        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)
        x = np.array(x)
        W1 = np.array(W1)
        b1 = np.array(b1)
        W2 = np.array(W2)
        y_true = np.array(y_true)

        ReLU = lambda x: (x>0) * x
        dReLU = lambda x: (x>0).astype('float')

        # sigmoid = lambda x : 1 / (1 + np.exp(-x))
        MSE = lambda y_hat, y_true: np.sum((y_hat-y_true)**2)/len(y_true)
        round_dict = lambda d, n: {key: np.round(d[key], n) for key in d}
        
        # dL_db = lambda y_hat, y_true: (y_hat - y_true)*y_hat*(1-y_hat)
        # dL_dw = lambda x, y_hat, y_true: (y_hat - y_true)*y_hat*(1-y_hat)*x

        z1 = np.dot(W1, x.T) + b1
        a1 = ReLU(z1)

        z2 = np.dot(W2, a1.T) + b2
        y_hat = ReLU(z2)

        L = np.mean((y_hat - y_true)**2)

        dL_dyhat = 2 * (y_hat - y_true) / y_hat.shape[0]

        delta2 = dL_dyhat * dReLU(z2)

        dW2 = np.outer(delta2, a1)
        db2 = delta2

        delta1 = np.outer(delta2.T, W2) * dReLU(z1)

        dW1 = np.outer(delta1.T, x)
        db1 = delta1
        ret = {
            'loss': L,
            'dW1': np.squeeze(dW1),
            'db1': np.squeeze(db1),
            'dW2': dW2,
            'db2': db2
        }
        return round_dict(ret, 4)