import numpy as np


def pixel_metrics(pred_masks, gt_masks):
    p = (np.asarray(pred_masks) > 0)
    g = (np.asarray(gt_masks) > 0)
    tp = float(np.sum(p & g))
    fp = float(np.sum(p & ~g))
    fn = float(np.sum(~p & g))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "iou": iou,
            "tp": tp, "fp": fp, "fn": fn}


def event_metrics(alert, gt_inzone):
    a = np.asarray(alert, dtype=bool)
    g = np.asarray(gt_inzone, dtype=bool)
    tp = float(np.sum(a & g))
    fp = float(np.sum(a & ~g))
    fn = float(np.sum(~a & g))
    tn = float(np.sum(~a & ~g))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / len(a) if len(a) > 0 else 0.0

    latencies = []
    n = len(g)
    j = 0
    while j < n:
        if g[j]:
            start = j
            while j < n and g[j]:
                j += 1
            seg = a[start:j]
            if seg.any():
                latencies.append(int(np.argmax(seg)))
            else:
                latencies.append(-1)
        else:
            j += 1
    hit = [l for l in latencies if l >= 0]
    mean_latency = float(np.mean(hit)) if hit else float("nan")
    detected = sum(1 for l in latencies if l >= 0)
    return {"precision": precision, "recall": recall, "f1": f1,
            "accuracy": accuracy, "mean_latency_frames": mean_latency,
            "events_total": len(latencies), "events_detected": detected}
