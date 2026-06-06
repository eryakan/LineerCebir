import numpy as np


def eigen_background(M, k=5):
    M = np.asarray(M, dtype=np.float64)
    m, n = M.shape
    k = int(min(k, n - 1, m))

    mean = M.mean(axis=1, keepdims=True)
    Xc = M - mean
    U, sigma, Vt = np.linalg.svd(Xc, full_matrices=False)
    Uk = U[:, :k]
    proj = Uk @ (Uk.T @ Xc)
    L = proj + mean
    S = M - L

    eigenvalues = (sigma ** 2) / max(n - 1, 1)
    energy = np.cumsum(eigenvalues) / np.sum(eigenvalues)

    info = {"eigenvalues": eigenvalues, "k": k, "energy": energy}
    return L, S, info
