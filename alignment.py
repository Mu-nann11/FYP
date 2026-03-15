import cv2
import numpy as np


def align_by_shift(dapi, target):
    if dapi.ndim == 3:
        dapi_gray = cv2.cvtColor(dapi, cv2.COLOR_BGR2GRAY)
    else:
        dapi_gray = dapi

    if target.ndim == 3:
        tgt_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    else:
        tgt_gray = target

    d = dapi_gray.astype(np.float32)
    t = tgt_gray.astype(np.float32)

    (dx, dy), _ = cv2.phaseCorrelate(t, d)

    max_shift = min(dapi_gray.shape[:2]) * 0.10
    if abs(dx) > max_shift or abs(dy) > max_shift:
        print("WARNING: estimated shift too large: dx=%.2f, dy=%.2f, use identity transform." % (dx, dy))
        dx, dy = 0.0, 0.0

    M = np.float32([[1, 0, dx], [0, 1, dy]])

    h, w = dapi_gray.shape[:2]
    aligned = cv2.warpAffine(
        target,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    if aligned.dtype != target.dtype:
        aligned = aligned.astype(target.dtype)

    return aligned, M, (dx, dy)
