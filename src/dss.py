import numpy as np
from .segmentation import blobs_per_frame


def _in_roi(centroid, roi):
    r, c = centroid
    r0, c0, r1, c1 = roi
    return (r0 <= r <= r1) and (c0 <= c <= c1)


def run_dss(masks, roi=None, min_blob_area=40, confirm_frames=3, clear_frames=3):
    H, W, n = masks.shape
    if roi is None:
        roi = (0, 0, H - 1, W - 1)
    frame_px = float(H * W)
    r0, c0, r1, c1 = roi
    roi_px = float(max((r1 - r0 + 1) * (c1 - c0 + 1), 1))

    candidate = np.zeros(n, dtype=bool)
    score = np.zeros(n, dtype=float)
    n_blobs = np.zeros(n, dtype=int)

    for j in range(n):
        m = masks[:, :, j]
        blobs = blobs_per_frame(m)
        n_blobs[j] = len(blobs)
        in_zone = False
        for b in blobs:
            if b["area"] >= min_blob_area and _in_roi(b["centroid"], roi):
                in_zone = True
                break
        candidate[j] = in_zone

        total_frac = m.sum() / frame_px
        roi_mask = m[r0:r1 + 1, c0:c1 + 1]
        zone_frac = roi_mask.sum() / roi_px
        score[j] = float(np.clip(0.35 * total_frac + 0.65 * zone_frac, 0.0, 1.0))

    alert = np.zeros(n, dtype=bool)
    state = False
    run_on = 0
    run_off = 0
    for j in range(n):
        if candidate[j]:
            run_on += 1
            run_off = 0
        else:
            run_off += 1
            run_on = 0
        if not state and run_on >= confirm_frames:
            state = True
        elif state and run_off >= clear_frames:
            state = False
        alert[j] = state

    events = []
    j = 0
    while j < n:
        if alert[j]:
            start = j
            while j < n and alert[j]:
                j += 1
            events.append((start, j - 1))
        else:
            j += 1

    return {
        "alert": alert,
        "candidate": candidate,
        "score": score,
        "n_blobs": n_blobs,
        "events": events,
        "roi": roi,
    }
