import unittest
from utils import batch_sqrtm
import numpy.testing as npt
import numpy as np
import cupy as cp
# import np

# sqrtm = MPA_Lya_2D.apply
# sqrtm_inv = MPA_Lya_Inv_2D.apply
# bsqrtm = MPA_Lya.apply # batch versions
# bsqrtm_inv = MPA_Lya_Inv.apply

rtol = 1e-4
atol = 1e-6
def bsqrtm_eigen(As):
    Rs = cp.zeros_like(As)
    N, n, _ = As.size()
    for i in range(N):
        Lambda, Omega = cp.linalg.eigh(As[i, :, :])
        Rs[i, :, : ] = Omega @ (Lambda.view(-1, 1).sqrt() * Omega.T)
    return Rs

def bsqrtm_eigen_inv(As):
    Rs = cp.zeros_like(As)
    N, n, _ = As.size()
    for i in range(N):
        Lambda, Omega = cp.linalg.eigh(As[i, :, :])
        Rs[i, :, : ] = Omega @ ((1/Lambda.view(-1, 1).sqrt()) * Omega.T)
    return Rs


class TestMatSqrt(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n = 50
        self.N = 100
        As = cp.random.randn(self.N, self.n, self.n)
        self.As = cp.matmul(As, cp.transpose(As, axes=[0, 2, 1]))

    def test_batch_mat_sqrt(self):
        # A = np_utils.
        Rs, Rsinv = batch_sqrtm(self.As)
        npt.assert_allclose(self.As.get(), cp.matmul(Rs, Rs).get(), rtol=rtol, atol=atol)
        npt.assert_allclose(self.As.get(), cp.linalg.inv(cp.matmul(Rsinv, Rsinv)).get(), rtol=rtol, atol=atol)

