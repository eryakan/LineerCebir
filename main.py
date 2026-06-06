import os
import json
import time
import numpy as np

from src.synth_data import make_synthetic_video, frames_to_matrix
from src.rpca import robust_pca
from src.pca_baseline import eigen_background
from src.segmentation import extract_masks
from src.dss import run_dss
from src.evaluate import pixel_metrics, event_metrics
from src import figures as F

ROOT = os.path.dirname(os.path.abspath(__file__))
RESDIR = os.path.join(ROOT, "results")
os.makedirs(RESDIR, exist_ok=True)


def main():
    np.random.seed(0)
    print("[1/6] Generating synthetic surveillance video ...")
    frames, gt_masks, roi, gt_inzone, meta = make_synthetic_video(
        H=120, W=160, n_frames=160, seed=0,
        illumination_drift=0.03, sensor_noise=0.006)
    H, W, n = frames.shape
    M = frames_to_matrix(frames)
    print(f"      frames={n}  size={H}x{W}  M shape={M.shape}  ROI={roi}")
    print(f"      ground-truth in-zone frames: {int(gt_inzone.sum())}/{n}")

    print("[2/6] Robust PCA (IALM) ...")
    lam = 0.5 / np.sqrt(max(M.shape))
    t0 = time.time()
    L, S, info = robust_pca(M, lam=lam, rho=1.5, tol=1e-6, max_iter=500, verbose=True)
    rpca_time = time.time() - t0
    print(f"      iters={info['iterations']}  rank(L)={info['rank']}  "
          f"converged={info['converged']}  time={rpca_time:.2f}s")

    print("[3/6] PCA eigen-background baseline ...")
    t0 = time.time()
    Lp, Sp, pinfo = eigen_background(M, k=5)
    pca_time = time.time() - t0
    print(f"      k={pinfo['k']}  variance@k={pinfo['energy'][pinfo['k']-1]:.3f}  "
          f"time={pca_time:.2f}s")

    print("[4/6] Foreground mask extraction ...")
    masks_rpca, thr_rpca = extract_masks(S, (H, W), min_area=20)
    masks_pca, thr_pca = extract_masks(Sp, (H, W), min_area=20)

    m_rpca = pixel_metrics(masks_rpca, gt_masks)
    m_pca = pixel_metrics(masks_pca, gt_masks)
    print(f"      RPCA pixel  P={m_rpca['precision']:.3f} R={m_rpca['recall']:.3f} "
          f"F1={m_rpca['f1']:.3f} IoU={m_rpca['iou']:.3f}")
    print(f"      PCA  pixel  P={m_pca['precision']:.3f} R={m_pca['recall']:.3f} "
          f"F1={m_pca['f1']:.3f} IoU={m_pca['iou']:.3f}")

    print("[5/6] Decision support system ...")
    dss = run_dss(masks_rpca, roi=roi, min_blob_area=30,
                  confirm_frames=3, clear_frames=4)
    em = event_metrics(dss["alert"], gt_inzone)
    print(f"      DSS frame P={em['precision']:.3f} R={em['recall']:.3f} "
          f"F1={em['f1']:.3f} acc={em['accuracy']:.3f}")
    print(f"      events detected {em['events_detected']}/{em['events_total']}  "
          f"mean latency={em['mean_latency_frames']:.1f} frames")
    print(f"      confirmed alert intervals: {dss['events']}")

    print("[6/6] Rendering figures ...")
    inzone_idx = np.where(gt_inzone)[0]
    idx = int(inzone_idx[len(inzone_idx) // 2]) if inzone_idx.size else n // 2
    bg_img = L[:, idx].reshape(H, W)
    fg_img = S[:, idx].reshape(H, W)
    paths = {}
    paths["architecture"] = F.fig_architecture()
    paths["decomposition"] = F.fig_decomposition_panel(
        frames[:, :, idx], bg_img, fg_img, masks_rpca[:, :, idx],
        gt_masks[:, :, idx], roi, idx)
    paths["spectrum"] = F.fig_singular_spectrum(info["sigma_M"], info["sigma_L"])
    paths["convergence"] = F.fig_convergence(info["errors"])
    paths["metrics_bar"] = F.fig_metrics_bar(m_rpca, m_pca)
    paths["dss_timeline"] = F.fig_dss_timeline(
        dss["score"], dss["alert"], gt_inzone, dss["events"])
    for k, v in paths.items():
        print(f"      figure[{k}] -> {os.path.relpath(v, ROOT)}")

    results = {
        "dataset": "synthetic_surveillance_demo",
        "meta": meta,
        "rpca": {
            "iterations": info["iterations"], "rank": info["rank"],
            "converged": info["converged"], "lambda": info["lambda"],
            "time_sec": rpca_time, "threshold": thr_rpca,
            "pixel_metrics": m_rpca,
        },
        "pca_baseline": {
            "k": pinfo["k"], "variance_at_k": float(pinfo["energy"][pinfo["k"] - 1]),
            "time_sec": pca_time, "threshold": thr_pca,
            "pixel_metrics": m_pca,
        },
        "dss": {
            "roi": list(roi), "event_metrics": em,
            "events": dss["events"],
            "n_inzone_gt_frames": int(gt_inzone.sum()),
        },
        "figures": {k: os.path.relpath(v, ROOT) for k, v in paths.items()},
    }
    with open(os.path.join(RESDIR, "metrics.json"), "w") as fh:
        json.dump(results, fh, indent=2, default=float)
    print("\nSaved results/metrics.json")
    return results


if __name__ == "__main__":
    main()
