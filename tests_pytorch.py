import unittest
from utils_pytorch import batch_sqrtm
import numpy.testing as npt
import numpy as np
# import cupy as torch
import torch
# import np

# sqrtm = MPA_Lya_2D.apply
# sqrtm_inv = MPA_Lya_Inv_2D.apply
# bsqrtm = MPA_Lya.apply # batch versions
# bsqrtm_inv = MPA_Lya_Inv.apply

rtol = 1e-4
atol = 1e-4
def bsqrtm_eigen(As):
    Rs = torch.zeros_like(As)
    N, n, _ = As.size()
    for i in range(N):
        Lambda, Omega = torch.linalg.eigh(As[i, :, :])
        Rs[i, :, : ] = Omega @ (Lambda.view(-1, 1).sqrt() * Omega.T)
    return Rs

def bsqrtm_eigen_inv(As):
    Rs = torch.zeros_like(As)
    N, n, _ = As.size()
    for i in range(N):
        Lambda, Omega = torch.linalg.eigh(As[i, :, :])
        Rs[i, :, : ] = Omega @ ((1/Lambda.view(-1, 1).sqrt()) * Omega.T)
    return Rs


class TestMatSqrt(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n = 50
        self.N = 100
        As = torch.randn(self.N, self.n, self.n)
        self.As = torch.matmul(As, As.transpose(1, 2))

    def test_batch_mat_sqrt(self):
        # A = np_utils.
        Rs, Rsinv = batch_sqrtm(self.As)
        npt.assert_allclose(self.As.numpy(), torch.matmul(Rs, Rs).numpy(), rtol=rtol, atol=atol)
        npt.assert_allclose(self.As.numpy(), torch.linalg.inv(torch.matmul(Rsinv, Rsinv)).numpy(), rtol=rtol, atol=atol)

