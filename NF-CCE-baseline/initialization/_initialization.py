# -*- coding: utf-8 -*-

import numpy as np
from math import ceil, sqrt
from operator import itemgetter
from numpy import linalg as la
from scipy import sparse


def _to_dense(X):
    """Ensure X is a dense ndarray."""
    if sparse.issparse(X):
        return X.toarray()
    return np.asarray(X)


def rnd_init(X, k):
    """
    Random initialization.
    """
    X = _to_dense(X)
    n = X.shape[0]
    H = np.random.rand(n, k)
    return H


def rnda_init(X, k, p=None):
    """
    RandomAcol initialization.
    """
    X = _to_dense(X)
    n = X.shape[0]

    if p is None:
        p = int(ceil(0.2 * n))

    prng = np.random.default_rng()
    H = np.zeros((n, k))

    for i in range(k):
        cols = prng.integers(low=0, high=n, size=p)
        H[:, i] = X[:, cols].mean(axis=1)

    return H


def rndc_init(X, k, p=None, l=None):
    """
    RandomC initialization.
    """
    X = _to_dense(X)
    n = X.shape[0]

    if p is None:
        p = int(ceil(0.2 * n))
    if l is None:
        l = int(ceil(0.5 * n))

    prng = np.random.default_rng()

    norms = [(i, la.norm(X[i, :], 2)) for i in range(n)]
    top = sorted(norms, key=itemgetter(1), reverse=True)[:l]
    top_idx = np.array([i for i, _ in top])

    H = np.zeros((n, k))
    for i in range(k):
        cols = prng.choice(top_idx, size=p, replace=True)
        H[:, i] = X[:, cols].mean(axis=1)

    return H


def svd_init(X, k, flag=0):
    """
    SVD-based initialization (Boutsidis & Gallopoulos, 2008).
    """
    X = _to_dense(X)
    n = X.shape[0]

    H = np.zeros((n, k))
    U, S, _ = la.svd(X, full_matrices=False)

    H[:, 0] = sqrt(S[0]) * np.abs(U[:, 0])

    for i in range(1, k):
        uu = U[:, i]
        uup = _pos(uu)
        n_uup = la.norm(uup, 2)
        if n_uup > 0:
            H[:, i] = sqrt(S[i]) * uup / n_uup

    H[H < 1e-10] = 0.0

    if flag in (1, 2):
        avg = X.mean()
        mask = (H == 0)
        if flag == 1:
            H[mask] = avg
        else:
            H[mask] = avg * np.random.random(mask.sum()) / 100.0

    return H


def _pos(x):
    """Return positive part of vector."""
    return np.maximum(x, 0)


def _neg(x):
    """Return negative part of vector."""
    return np.maximum(-x, 0)


