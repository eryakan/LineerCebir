import os
import glob
import numpy as np
from PIL import Image


def _read_gray(path, size=None):
    im = Image.open(path).convert("L")
    if size is not None:
        im = im.resize((size[1], size[0]), Image.BILINEAR)
    return np.asarray(im, dtype=np.float64) / 255.0


def load_cdnet_sequence(seq_dir, resize=None, max_frames=None):
    in_dir = os.path.join(seq_dir, "input")
    gt_dir = os.path.join(seq_dir, "groundtruth")

    def _first_match(d, patterns):
        for p in patterns:
            files = sorted(glob.glob(os.path.join(d, p)))
            if files:
                return files
        return []

    in_files = _first_match(in_dir, ["in*.jpg", "in*.png", "*.jpg", "*.png"])
    gt_files = _first_match(gt_dir, ["gt*.png", "gt*.bmp", "*.png", "*.bmp"])
    if not in_files:
        raise FileNotFoundError(
            f"No input frames in {in_dir}. Point seq_dir at a leaf sequence "
            f"folder that directly contains 'input/' and 'groundtruth/'.")

    t0, t1 = 1, len(in_files)
    troi = os.path.join(seq_dir, "temporalROI.txt")
    if os.path.exists(troi):
        with open(troi) as fh:
            parts = fh.read().split()
            if len(parts) >= 2:
                t0, t1 = int(parts[0]), int(parts[1])

    idx = list(range(t0 - 1, min(t1, len(in_files))))
    if max_frames is not None:
        idx = idx[:max_frames]

    first = _read_gray(in_files[idx[0]], resize)
    H, W = first.shape
    n = len(idx)
    frames = np.zeros((H, W, n), dtype=np.float64)
    gt_masks = np.zeros((H, W, n), dtype=np.uint8)
    valid = np.ones((H, W, n), dtype=bool)

    have_gt = len(gt_files) >= len(in_files)
    for k, i in enumerate(idx):
        frames[:, :, k] = _read_gray(in_files[i], resize)
        if have_gt:
            g = Image.open(gt_files[i]).convert("L")
            if resize is not None:
                g = g.resize((W, H), Image.NEAREST)
            g = np.asarray(g)
            gt_masks[:, :, k] = (g == 255).astype(np.uint8)
            valid[:, :, k] = ~np.isin(g, [85, 170])

    roi_path = os.path.join(seq_dir, "ROI.bmp")
    if os.path.exists(roi_path):
        roi_im = Image.open(roi_path).convert("L")
        if resize is not None:
            roi_im = roi_im.resize((W, H), Image.NEAREST)
        roi = np.asarray(roi_im) > 127
        valid &= roi[:, :, None]

    meta = {"H": H, "W": W, "n_frames": n, "t0": t0, "t1": t1,
            "sequence": os.path.basename(os.path.normpath(seq_dir))}
    return frames, gt_masks, valid, meta
