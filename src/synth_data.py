import numpy as np


def _static_background(H, W, rng):
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    bg = 0.45 + 0.15 * np.sin(2 * np.pi * xx / W) + 0.10 * np.cos(2 * np.pi * yy / H)
    bg[int(H * 0.55):int(H * 0.70), :] *= 0.75
    bg[int(H * 0.15):int(H * 0.45), int(W * 0.05):int(W * 0.25)] *= 1.15
    bg += 0.02 * rng.standard_normal((H, W))
    return np.clip(bg, 0, 1)


def make_synthetic_video(H=120, W=160, n_frames=160, seed=0,
                         illumination_drift=0.04, sensor_noise=0.015):
    rng = np.random.default_rng(seed)
    bg = _static_background(H, W, rng)

    roi = (int(H * 0.30), int(W * 0.55), int(H * 0.85), int(W * 0.92))

    frames = np.zeros((H, W, n_frames), dtype=np.float64)
    gt_masks = np.zeros((H, W, n_frames), dtype=np.uint8)
    gt_inzone = np.zeros(n_frames, dtype=bool)

    def linear_path(p0, p1, t0, t1):
        def f(t):
            if t < t0 or t > t1:
                return None
            a = (t - t0) / max(t1 - t0, 1)
            r = (1 - a) * p0[0] + a * p1[0]
            c = (1 - a) * p0[1] + a * p1[1]
            return (r, c)
        return f

    objects = [
        dict(kind="rect", h=10, w=18, intensity=0.05,
             path=linear_path((H * 0.62, -20), (H * 0.62, W + 20), 10, 110)),
        dict(kind="circle", rad=5, intensity=0.9,
             path=linear_path((-10, W * 0.75), (H + 10, W * 0.75), 40, 150)),
        dict(kind="circle", rad=4, intensity=0.95,
             path=linear_path((H * 0.10, W * 0.05), (H * 0.10, W * 0.45), 20, 90)),
    ]

    for t in range(n_frames):
        frame = bg.copy()
        frame = np.clip(frame + illumination_drift * np.sin(2 * np.pi * t / n_frames), 0, 1)

        mask = np.zeros((H, W), dtype=np.uint8)
        any_in_zone = False
        for ob in objects:
            pos = ob["path"](t)
            if pos is None:
                continue
            r, c = pos
            if ob["kind"] == "rect":
                h, w = ob["h"], ob["w"]
                r0 = int(r - h / 2); r1 = int(r + h / 2)
                c0 = int(c - w / 2); c1 = int(c + w / 2)
                rr0, rr1 = max(r0, 0), min(r1, H)
                cc0, cc1 = max(c0, 0), min(c1, W)
                if rr1 > rr0 and cc1 > cc0:
                    frame[rr0:rr1, cc0:cc1] = np.clip(
                        frame[rr0:rr1, cc0:cc1] * 0 + ob["intensity"], 0, 1)
                    mask[rr0:rr1, cc0:cc1] = 1
            else:
                rad = ob["rad"]
                yy, xx = np.mgrid[0:H, 0:W]
                disk = (yy - r) ** 2 + (xx - c) ** 2 <= rad ** 2
                frame[disk] = ob["intensity"]
                mask[disk] = 1

            rr0, cc0, rr1, cc1 = roi
            if (rr0 <= r <= rr1) and (cc0 <= c <= cc1):
                any_in_zone = True

        frame = np.clip(frame + sensor_noise * rng.standard_normal((H, W)), 0, 1)
        frames[:, :, t] = frame
        gt_masks[:, :, t] = mask
        gt_inzone[t] = any_in_zone

    meta = {"H": H, "W": W, "n_frames": n_frames, "seed": seed, "roi": roi,
            "n_objects": len(objects)}
    return frames, gt_masks, roi, gt_inzone, meta


def frames_to_matrix(frames):
    H, W, n = frames.shape
    return frames.reshape(H * W, n)
