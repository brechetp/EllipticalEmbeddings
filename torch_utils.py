#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 10:23:43 2018

@author: boris
"""

import numpy as np
# import cupy as torch
import torch

def wishart(n_points, dim=2, p=5):
    """
    Wishart sampling
    """
    X = torch.randn(n_points, dim, p)
    return torch.matmul(X, torch.transpose(X, 2, 1))


def batch_sqrtm(A, numIters = 200, reg = 1.0):
    """
    Batch matrix root via Newton-Schulz iterations
    """

    batchSize = A.shape[0]
    dim = A.shape[1]
    device = A.device
    #Renormalize so that the each matrix has a norm lesser than 1/reg, but only normalize when necessary
    normA = reg * torch.linalg.norm(A, dim=(1, 2))
    renorm_factor = torch.ones_like(normA)
    renorm_factor[torch.where(normA > 1.0)] = normA[torch.where(normA > 1.0)]
    renorm_factor = renorm_factor.reshape(batchSize, 1, 1)

    Y = torch.divide(A, renorm_factor)
    I = torch.eye(dim).reshape(1, dim, dim).repeat(batchSize, 1, 1).to(device)
    Z = torch.eye(dim).reshape(1, dim, dim).repeat(batchSize, 1, 1).to(device)
    for i in range(numIters):
        T = 0.5 * (3.0 * I - torch.matmul(Z, Y))
        Y = torch.matmul(Y, T)
        Z = torch.matmul(T, Z)
    sA = Y * torch.sqrt(renorm_factor)
    sAinv = Z / torch.sqrt(renorm_factor)
    return sA, sAinv


def batch_bures(U, V, numIters = 20, U_stride=None, sU = None, inv_sU = None, prod = False):
    #Avoid recomputing roots if not necessary
    if sU is None:
        if U_stride is not None:
            sU_, inv_sU_ = batch_sqrtm(U[::U_stride], numIters=numIters)
            sU = sU_.repeat(U_stride, 1, 1)
            inv_sU = inv_sU_.repeat(U_stride, 1, 1)
        else :
            sU, inv_sU = batch_sqrtm(U, numIters=numIters)
    cross, inv = batch_sqrtm(torch.matmul(sU, torch.matmul(V, sU)), numIters = numIters)
    if prod:
        return torch.trace(cross, dim1=1, dim2=2), inv, sU, inv_sU, cross
    else:
        return torch.trace(U + V - 2 * cross, dim1=1, dim2=2), inv, sU, inv_sU, cross


def batch_W2(m1, m2, U, V, Cn = 1, numIters = 20, U_stride=None, sU = None, inv_sU = None, prod = False):
    """
    Squared Wasserstein distance between N(m1, U) and N(m2, V)
    """
    bb, inv, sU, inv_sU, mid = batch_bures(U, V, numIters = numIters, U_stride=U_stride, sU = sU, inv_sU = inv_sU, prod=prod)
    if prod:
        return (m1*m2).sum(dim=1) + Cn * bb, inv, sU, inv_sU, mid
    else:
        return ((m1 - m2)**2).sum(dim=1) + Cn * bb, inv, sU, inv_sU, mid

def batch_Tuv(U, V, inv=None, sV=None, numIters = 2):
    """
    Returns the transportation matrix from N(U) to N(V):
    V^{1/2}[V^{1/2}UV^{1/2}]^{-1/2}V^{1/2}
    """
    if sV is None:
        sV, _ = batch_sqrtm(V, numIters=numIters)
    if inv is None:
        _, inv = batch_sqrtm(torch.matmul(torch.matmul(sV, U), sV), numIters = numIters)
    return torch.matmul(sV, torch.matmul(inv, sV))

def batch_log(U, V, inv=None, sV=None, numIters = 2, prod = False):
    """
    Log map at N(U) of N(V)
    """
    batchsize = U.shape[0]
    n = U.shape[1]
    if prod:
        return batch_Tuv(U, V, inv, sV, numIters=numIters)
    else:
        return batch_Tuv(U, V, inv, sV, numIters = numIters) - torch.eye(n).reshape(1, n, n).repeat(batchsize, 1, 1)

def batch_Tuv2(U, V, mid=None, inv_sU=None, numIters = 2):
    """
    Returns the transportation matrix from N(U) to N(V):
    V^{-1/2}[V^{1/2}UV^{1/2}]^{1/2}V^{-1/2}
    """
    if (inv_sU is None) or (mid is None):
        sU, inv_sU = batch_sqrtm(U, numIters = numIters)
    if mid is None:
        mid, _ = batch_sqrtm(torch.matmul(torch.matmul(sU, V), sU), numIters = numIters)
    return torch.matmul(inv_sU, torch.matmul(mid, inv_sU))

def batch_log2(U, V, mid=None, inv_sU=None, numIters = 2, prod = False):
    """
    Log map at N(U) of N(V)
    """
    batchsize = U.shape[0]
    n = U.shape[1]
    if prod:
        return batch_Tuv2(U, V, mid, inv_sU, numIters=numIters)
    else:
        return batch_Tuv2(U, V, mid, inv_sU, numIters = numIters) - torch.eye(n).reshape(1, n, n).repeat(batchsize, 1, 1)

def batch_exp(U, V):
    """
    Exponential map at N(U) in the direction of V
    """
    batchsize = U.shape[0]
    n = V.shape[1]
    V_I = V + torch.eye(n).reshape(1, n, n).repeat(batchsize, 1, 1)
    return torch.matmul(V_I, torch.matmul(U, V_I))


def to_full(L):
    xp = torch.get_array_module(L)
    return xp.matmul(L, xp.transpose(L, 2, 1))

def diag_bures(U, V):
    """
    Batched squared bures distance between diagonal covariances U and V, represented as batch of vectors
    """
    return ((torch.sqrt(U) - torch.sqrt(V))**2).sum(dim=1)

def diag_W2(m1, m2, U, V, Cn = 1):
    """
    Squared Wasserstein distance between E(m1, U) and E(m2, V), where U and V are diagonal matrices
    """
    return ((m1 - m2)**2).sum(dim=1) + Cn * diag_bures(U, V)

def bures_cosine(m1, m2, U, V, Cn = 1, numIters = 20):
    bb = batch_bures(U, V, numIters = numIters, prod=True)[0]

    return ((m1*m2).sum(dim=1) + Cn * bb) / torch.sqrt((torch.linalg.norm(m1, dim=1)**2 + Cn * torch.trace(U, dim1=1, dim2=2) + 1E-8) * (torch.linalg.norm(m2, dim=1)**2 + Cn * torch.trace(V, dim1=1, dim2=2) + 1E-8))


def sum_cosine(m1, m2, U, V, Cn = 1, numIters = 20):

    bb = batch_bures(U, V, numIters = numIters, prod=True)[0]
    #print torch.mean(bb)
    #print torch.mean((m1*m2).sum(dim=1))

    return (m1*m2).sum(dim=1) / (torch.linalg.norm(m1, dim=1) * torch.linalg.norm(m2, dim=1) + 1E-8)\
           + Cn * bb / torch.sqrt((torch.trace(U, dim1=1, dim2=2) * torch.trace(V, dim1=1, dim2=2)) + 1E-8)

def sum_by_group(values1, values2, groups):
    order = torch.argsort(groups)
    groups = groups[order]
    values1 = values1[order]
    values2 = values2[order]
    torch.cumsum(values1, out=values1, dim=0)
    torch.cumsum(values2, out=values2, dim=0)
    index = torch.ones(len(groups), 'bool')
    index[:-1] = groups[1:] != groups[:-1]
    values1 = values1[index]
    values2 = values2[index]
    groups = groups[index]
    values1[1:] = values1[1:] - values1[:-1]
    values2[1:] = values2[1:] - values2[:-1]
    return values1, values2, groups


def symmetrize(M):
    return (M + torch.transpose(M, 2, 1)) / 2.0
