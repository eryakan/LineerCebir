import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

FIGDIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "figures")
os.makedirs(FIGDIR, exist_ok=True)


def _save(fig, name):
    path = os.path.join(FIGDIR, name)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def fig_decomposition_panel(frame, bg, fg, mask, gt, roi, idx, name="fig_decomposition.png"):
    fig, axes = plt.subplots(1, 5, figsize=(13, 3))
    titles = ["(a) Girdi karesi", "(b) Arka plan L\n(dusuk kademe)",
              "(c) Sparse |S|\n(on plan)", "(d) Tahmin maskesi", "(e) Yer-gercegi"]
    imgs = [frame, bg, np.abs(fg), mask, gt]
    cmaps = ["gray", "gray", "magma", "gray", "gray"]
    for ax, im, t, cm in zip(axes, imgs, titles, cmaps):
        ax.imshow(im, cmap=cm)
        ax.set_title(t, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    r0, c0, r1, c1 = roi
    for ax in (axes[0], axes[3], axes[4]):
        ax.add_patch(Rectangle((c0, r0), c1 - c0, r1 - r0, fill=False,
                               edgecolor="cyan", lw=1.5, ls="--"))
    fig.suptitle(f"Kare #{idx}: RPCA ayrisimi (kesikli mavi = kisitli bolge)", y=1.04, fontsize=10)
    return _save(fig, name)


def fig_singular_spectrum(sigma_M, sigma_L, name="fig_singular_spectrum.png"):
    fig, ax = plt.subplots(figsize=(6, 4))
    k = min(60, len(sigma_M))
    ax.semilogy(np.arange(1, k + 1), sigma_M[:k], "o-", ms=3, label="M (girdi matrisi)")
    ax.semilogy(np.arange(1, len(sigma_L[:k]) + 1), sigma_L[:k], "s-", ms=3,
                label="L (dusuk kademe bilesen)")
    ax.set_xlabel("Tekil deger indeksi")
    ax.set_ylabel("Tekil deger (log)")
    ax.set_title("Tekil deger spektrumu: arka planin dusuk kademeli yapisi")
    ax.legend()
    return _save(fig, name)


def fig_convergence(errors, name="fig_convergence.png"):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogy(np.arange(1, len(errors) + 1), errors, "-", lw=1.6)
    ax.set_xlabel("Iterasyon")
    ax.set_ylabel(r"$\|M-L-S\|_F / \|M\|_F$ (log)")
    ax.set_title("IALM yakinsama egrisi")
    return _save(fig, name)


def fig_metrics_bar(rpca_m, pca_m, name="fig_metrics_bar.png"):
    keys = ["precision", "recall", "f1", "iou"]
    labels = ["Precision", "Recall", "F1", "IoU"]
    x = np.arange(len(keys)); w = 0.38
    fig, ax = plt.subplots(figsize=(6.5, 4))
    b1 = ax.bar(x - w / 2, [rpca_m[k] for k in keys], w, label="Robust PCA")
    b2 = ax.bar(x + w / 2, [pca_m[k] for k in keys], w, label="PCA eigen-background")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05); ax.set_ylabel("Skor")
    ax.set_title("Piksel duzeyi segmentasyon basarisi")
    ax.legend()
    for b in list(b1) + list(b2):
        ax.annotate(f"{b.get_height():.2f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                    ha="center", va="bottom", fontsize=8)
    return _save(fig, name)


def fig_dss_timeline(score, alert, gt_inzone, events, name="fig_dss_timeline.png"):
    n = len(score)
    fig, ax = plt.subplots(figsize=(9, 3.4))
    t = np.arange(n)
    ax.plot(t, score, color="#333", lw=1.2, label="Anomali skoru")
    ax.fill_between(t, 0, 1, where=gt_inzone, color="orange", alpha=0.18,
                    label="Yer-gercegi izinsiz giris")
    ax.fill_between(t, 0, 1, where=alert, color="red", alpha=0.0)
    for (s, e) in events:
        ax.axvspan(s, e, color="red", alpha=0.22)
    ax.plot([], [], color="red", alpha=0.4, lw=8, label="Onaylanmis alarm")
    ax.set_xlabel("Kare indeksi"); ax.set_ylabel("Skor")
    ax.set_ylim(0, max(0.12, float(np.max(score)) * 1.4))
    ax.set_title("KDS zaman cizelgesi: skor, yer-gercegi ve onaylanmis alarmlar")
    ax.legend(loc="upper right", fontsize=8, ncol=3)
    return _save(fig, name)


def fig_architecture(name="fig_architecture.png"):
    fig, ax = plt.subplots(figsize=(11, 2.8))
    ax.set_xlim(0, 11.4); ax.set_ylim(0, 3); ax.axis("off")
    boxes = [
        (0.2, "Video\n(kareler)"),
        (1.85, "On isleme\ngri + olcekleme\nM matrisi"),
        (3.7, "Robust PCA\nL (dusuk kademe)\n+ S (sparse)\n[SVD]"),
        (5.75, "Maske cikarimi\nesik + morfoloji\n+ bilesen analizi"),
        (7.75, "Karar Destek\nROI + kurallar\n+ zamansal onay"),
        (9.7, "Cikti\nalarm + skor\n+ metrikler"),
    ]
    w, h, y = 1.5, 1.7, 0.65
    for x, label in boxes:
        ax.add_patch(Rectangle((x, y), w, h, fc="#e8eef7", ec="#2E5A8A", lw=1.6))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=8.5)
    for i in range(len(boxes) - 1):
        x0 = boxes[i][0] + w
        x1 = boxes[i + 1][0]
        ax.add_patch(FancyArrowPatch((x0, y + h / 2), (x1, y + h / 2),
                     arrowstyle="-|>", mutation_scale=14, lw=1.4, color="#2E5A8A"))
    ax.text(5.5, 2.75, "Sistem mimarisi", ha="center", fontsize=11, weight="bold")
    return _save(fig, name)
