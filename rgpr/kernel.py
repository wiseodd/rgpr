import torch
import torch.nn.functional as F
from gpytorch import kernels
import math


def k_cubic_spline(x1, x2, var=1, c=0):
    min = torch.min(x1, x2)
    return var * (1/3*(min**3-c**3) - 1/2*(min**2-c**2)*(x1+x2) + (min-c)*x1*x2)


def gamma(x):
    return 0.5*(torch.sign(x)+1)


def kernel_1d(x1, x2, var=1):
    pos_val = k_cubic_spline(x1, x2, var)
    neg_val = k_cubic_spline(-x1, -x2, var)
    return gamma(x1)*gamma(x2)*pos_val + gamma(-x1)*gamma(-x2)*neg_val


def kernel(x1, x2, var=1):
    assert x1.shape == x2.shape

    orig_shape = x1.shape

    x1, x2 = x1.reshape(-1, 1), x2.reshape(-1, 1)
    k = kernel_1d(x1, x2, var).reshape(orig_shape)
    out = k.mean(-1)

    return out


class DSCSKernel(kernels.Kernel):

    def __init__(self, var=1):
        super().__init__()
        self.var = var

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        n, m = len(x1), len(x2)

        # For broadcasting
        x1 = x1.unsqueeze(1)  # (n, 1, k)
        x2 = x2.unsqueeze(0)  # (1, m, k)

        K = kernel_1d(x1, x2, self.var).mean(-1)

        return K

    def is_stationary():
        return False
