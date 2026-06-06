import os
import json
import time
import argparse
import numpy as np

from src.cdnet_loader import load_cdnet_sequence
from src.synth_data import frames_to_matrix
from src.rpca import robust_pca
from src.pca_baseline import eigen_background
from src.segmentation import extract_masks
from src.dss import run_dss
from src.evaluate import pixel_metrics, event_metrics
from src import figures as F

ROOT = os.path.dirname(os.path.abspath(__file__))
RESDIR = os.path.join(ROOT, "results")
os.makedirs(RESDIR, exist_ok=True)


def parse_args():
    p = argparse.ArgumentParser(description="Run the pipeline on a CDnet 2014 sequence.")
    p.add_argument("--seq", default=os.path.join(ROOT, "data", "highway"),
                   help="Leaf sequence folder containing input/ and groundtruth/.")
    p.add_argument("--resize", nargs=2, type=int, default=[120, 160],
                   metavar=("H", "W"), help="Downscale size for tractable SVD.")
    p.add_argument("--max-frames", type=int, default=300,
                   help="Cap on number of frames (memory/time).")
    p.add_argument("--lam-mult", type=float, default=0.5,
                   help="Multiplier on the standard lambda = 1/sqrt(max(m,n)).")
    p.add_argument("--pca-k", type=int, default=5, help="PCA component count.")
    p.add_argument("--zone", nargs=4, type=float,
                   default=[0.35, 0.30, 0.95, 0.75],
                   metavar=("r0", "c0", "r1", "c1"),
                   help="Restricted zone as fractions of (H,W): r0 c0 r1 c1.")
    p.add_argument("--zone-area", type=int, default=20,
                   help="Min in-zone GT foreground pixels to call a frame an intrusion.")
    return p.parse_args()


def main():
    a = parse_args()
    print("[1/6] Loading CDnet sequence:", a.seq)
    resize = tuple(a.resize) if a.resize else None
    frames, gt_masks, valid, meta = load_cdnet_sequence(
        a.seq, resize=resize, max_frames=a.max_frames)
    H, W, n = frames.shape
    M = frames_to_matrix(frames)
    print(f"      sequence={meta['sequence']}  frames={n}  size={H}x{W}  M={M.shape}")

    zr0 = int(a.zone[0] * H); zc0 = int(a.zone[1] * W)
    zr1 = int(a.zone[2] * H); zc1 = int(a.zone[3] * W)
    zone = (zr0, zc0, zr1, zc1)
    zone_gt = gt_masks[zr0:zr1 + 1, zc0:zc1 + 1, :].reshape(-1, n).sum(axis=0)
    gt_inzone = zone_gt >= a.zone_area
    print(f"      zone={zone}  GT in-zone frames={int(gt_inzone.sum())}/{n}")

    print("[2/6] Robust PCA (IALM) ...")
    lam = a.lam_mult / np.sqrt(max(M.shape))
    t0 = time.time()
    L, S, info = robust_pca(M, lam=lam, rho=1.5, tol=1e-6, max_iter=500, verbose=True)
    rpca_time = time.time() - t0
    print(f"      iters={info['iterations']}  rank(L)={info['rank']}  "
          f"converged={info['converged']}  time={rpca_time:.2f}s")

    print("[3/6] PCA eigen-background baseline ...")
    t0 = time.time()
    Lp, Sp, pinfo = eigen_background(M, k=a.pca_k)
    pca_time = time.time() - t0

    print("[4/6] Foreground mask extraction ...")
    masks_rpca, thr_rpca = extract_masks(S, (H, W), min_area=20)
    masks_pca, thr_pca = extract_masks(Sp, (H, W), min_area=20)

    vb = valid.astype(bool)
    gtb = (gt_masks > 0) & vb
    m_rpca = pixel_metrics((masks_rpca > 0) & vb, gtb)
    m_pca = pixel_metrics((masks_pca > 0) & vb, gtb)
    print(f"      RPCA pixel  P={m_rpca['precision']:.3f} R={m_rpca['recall']:.3f} "
          f"F1={m_rpca['f1']:.3f} IoU={m_rpca['iou']:.3f}")
    print(f"      PCA  pixel  P={m_pca['precision']:.3f} R={m_pca['recall']:.3f} "
          f"F1={m_pca['f1']:.3f} IoU={m_pca['iou']:.3f}")

    print("[5/6] Decision support system ...")
    dss = run_dss(masks_rpca, roi=zone, min_blob_area=25,
                  confirm_frames=3, clear_frames=4)
    em = event_metrics(dss["alert"], gt_inzone)
    print(f"      DSS frame P={em['precision']:.3f} R={em['recall']:.3f} "
          f"F1={em['f1']:.3f} acc={em['accuracy']:.3f}")
    print(f"      events detected {em['events_detected']}/{em['events_total']}  "
          f"mean latency={em['mean_latency_frames']}")

    print("[6/6] Rendering figures (suffix _cdnet) ...")
    inzone_idx = np.where(gt_inzone)[0]
    idx = int(inzone_idx[len(inzone_idx) // 2]) if inzone_idx.size else n // 2
    paths = {}
    paths["decomposition"] = F.fig_decomposition_panel(
        frames[:, :, idx], L[:, idx].reshape(H, W), S[:, idx].reshape(H, W),
        masks_rpca[:, :, idx], gt_masks[:, :, idx], zone, idx,
        name="fig_decomposition_cdnet.png")
    paths["spectrum"] = F.fig_singular_spectrum(
        info["sigma_M"], info["sigma_L"], name="fig_singular_spectrum_cdnet.png")
    paths["convergence"] = F.fig_convergence(info["errors"], name="fig_convergence_cdnet.png")
    paths["metrics_bar"] = F.fig_metrics_bar(m_rpca, m_pca, name="fig_metrics_bar_cdnet.png")
    paths["dss_timeline"] = F.fig_dss_timeline(
        dss["score"], dss["alert"], gt_inzone, dss["events"], name="fig_dss_timeline_cdnet.png")
    for k, v in paths.items():
        print(f"      figure[{k}] -> {os.path.relpath(v, ROOT)}")

    results = {
        "dataset": "CDnet2014", "sequence": meta["sequence"], "meta": meta,
        "resize": list(resize) if resize else None, "max_frames": a.max_frames,
        "rpca": {"iterations": info["iterations"], "rank": info["rank"],
                 "converged": info["converged"], "lambda": info["lambda"],
                 "time_sec": rpca_time, "pixel_metrics": m_rpca},
        "pca_baseline": {"k": pinfo["k"],
                         "variance_at_k": float(pinfo["energy"][pinfo["k"] - 1]),
                         "time_sec": pca_time, "pixel_metrics": m_pca},
        "dss": {"zone": list(zone), "zone_area_thresh": a.zone_area,
                "event_metrics": em, "events": dss["events"],
                "n_inzone_gt_frames": int(gt_inzone.sum())},
        "figures": {k: os.path.relpath(v, ROOT) for k, v in paths.items()},
    }
    out = os.path.join(RESDIR, "metrics_cdnet.json")
    with open(out, "w") as fh:
        json.dump(results, fh, indent=2, default=float)
    print("\nSaved", os.path.relpath(out, ROOT))
    return results


if __name__ == "__main__":
    main()
