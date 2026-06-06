import numpy as np
from scipy import ndimage as ndi


def _otsu_threshold(values):
    v = values[np.isfinite(values)]
    if v.size == 0:
        return 0.0
    hist, edges = np.histogram(v, bins=256)
    centers = (edges[:-1] + edges[1:]) / 2.0
    total = hist.sum()
    if total == 0:
        return 0.0
    w = np.cumsum(hist)
    mu = np.cumsum(hist * centers)
    mu_t = mu[-1]
    denom = w * (total - w)
    denom[denom == 0] = 1
    sigma_b2 = (mu_t * w - mu) ** 2 / denom
    idx = int(np.argmax(sigma_b2))
    return centers[idx]


def extract_masks(S, frame_shape, threshold=None, k_sigma=4.0, floor=0.02,
                  open_size=2, close_size=3, min_area=20):
    H, W = frame_shape
    m, n = S.shape
    mag = np.abs(S)

    if threshold is None:
        med = np.median(mag)
        mad = np.median(np.abs(mag - med))
        robust_std = 1.4826 * mad
        threshold = max(med + k_sigma * robust_std, floor)

    open_st = np.ones((open_size, open_size), dtype=bool) if open_size > 0 else None
    close_st = np.ones((close_size, close_size), dtype=bool) if close_size > 0 else None

    masks = np.zeros((H, W, n), dtype=np.uint8)
    for j in range(n):
        bw = (mag[:, j].reshape(H, W) > threshold)
        if open_st is not None:
            bw = ndi.binary_opening(bw, structure=open_st)
        if close_st is not None:
            bw = ndi.binary_closing(bw, structure=close_st)
        if min_area > 0 and bw.any():
            lab, num = ndi.label(bw)
            if num > 0:
                sizes = ndi.sum(np.ones_like(lab), lab, index=np.arange(1, num + 1))
                keep = np.where(sizes >= min_area)[0] + 1
                bw = np.isin(lab, keep)
        masks[:, :, j] = bw.astype(np.uint8)
    return masks, float(threshold)


def blobs_per_frame(mask):
    lab, num = ndi.label(mask)
    out = []
    if num == 0:
        return out
    for i in range(1, num + 1):
        ys, xs = np.where(lab == i)
        if ys.size == 0:
            continue
        out.append({
            "area": int(ys.size),
            "centroid": (float(ys.mean()), float(xs.mean())),
            "bbox": (int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max())),
        })
    return out
