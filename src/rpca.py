import numpy as np


def _svd_threshold(X, tau):
    U, sigma, Vt = np.linalg.svd(X, full_matrices=False)
    sigma_shrunk = np.maximum(sigma - tau, 0.0)
    rank = int(np.sum(sigma_shrunk > 0))
    L = (U * sigma_shrunk) @ Vt
    return L, rank, sigma


def _soft_threshold(X, tau):
    return np.sign(X) * np.maximum(np.abs(X) - tau, 0.0)


def robust_pca(M, lam=None, mu=None, rho=1.5, tol=1e-7, max_iter=500, verbose=False):
    M = np.asarray(M, dtype=np.float64)
    m, n = M.shape

    if lam is None:
        lam = 1.0 / np.sqrt(max(m, n))

    norm_fro = np.linalg.norm(M, "fro")
    norm_two = np.linalg.norm(M, 2)
    norm_inf = np.linalg.norm(M.ravel(), np.inf) / lam
    dual_norm = max(norm_two, norm_inf)

    Y = M / dual_norm
    S = np.zeros_like(M)
    L = np.zeros_like(M)

    if mu is None:
        mu = 1.25 / norm_two
    mu_max = mu * 1e7

    errors = []
    converged = False
    rank = 0
    sigma_M = np.linalg.svd(M, compute_uv=False)
    sigma_L = sigma_M.copy()

    for it in range(1, max_iter + 1):
        L, rank, _ = _svd_threshold(M - S + Y / mu, 1.0 / mu)
        S = _soft_threshold(M - L + Y / mu, lam / mu)
        residual = M - L - S
        Y = Y + mu * residual
        mu = min(rho * mu, mu_max)

        err = np.linalg.norm(residual, "fro") / norm_fro
        errors.append(err)
        if verbose and (it % 10 == 0 or it == 1):
            print(f"  iter {it:4d}  rank(L)={rank:4d}  err={err:.3e}")
        if err < tol:
            converged = True
            break

    sigma_L = np.linalg.svd(L, compute_uv=False)

    info = {
        "iterations": it,
        "rank": rank,
        "converged": converged,
        "errors": errors,
        "lambda": lam,
        "sigma_M": sigma_M,
        "sigma_L": sigma_L,
    }
    return L, S, info
