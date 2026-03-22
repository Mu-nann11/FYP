import cv2
import numpy as np
from typing import Tuple, Optional


def _to_gray(img: np.ndarray) -> np.ndarray:
    """Convert to single-channel float32 grayscale."""
    if img.ndim == 3:
        if img.shape[2] == 4:
            gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    return gray.astype(np.float32)


def _warp(target: np.ndarray, M: np.ndarray, h: int, w: int) -> np.ndarray:
    """Apply affine/perspective transform, preserve original dtype."""
    if M.shape == (3, 3):
        aligned = cv2.warpPerspective(target, M, (w, h),
                                       flags=cv2.INTER_LINEAR,
                                       borderMode=cv2.BORDER_CONSTANT,
                                       borderValue=0)
    else:
        aligned = cv2.warpAffine(target, M, (w, h),
                                  flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=0)
    if aligned.dtype != target.dtype:
        aligned = aligned.astype(target.dtype)
    return aligned


# ── Quality metrics ──────────────────────────────────────────────

def mutual_information(img1: np.ndarray, img2: np.ndarray, bins: int = 64) -> float:
    """
    Compute normalized mutual information between two images.
    Higher = better alignment. Range [0, 1].
    """
    a = _to_gray(img1).ravel()
    b = _to_gray(img2).ravel()

    # Normalize to [0, bins-1]
    a = ((a - a.min()) / (a.max() - a.min() + 1e-8) * (bins - 1)).astype(np.int32)
    b = ((b - b.min()) / (b.max() - b.min() + 1e-8) * (bins - 1)).astype(np.int32)

    hist_2d, _, _ = np.histogram2d(a, b, bins=bins)
    pxy = hist_2d / (hist_2d.sum() + 1e-8)
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)

    # Mutual information
    nz = pxy > 0
    mi = np.sum(pxy[nz] * np.log(pxy[nz] / (px[:, None] * py[None, :])[nz] + 1e-8))

    # Normalized MI
    hx = -np.sum(px[px > 0] * np.log(px[px > 0]))
    hy = -np.sum(py[py > 0] * np.log(py[py > 0]))
    nmi = 2 * mi / (hx + hy + 1e-8)
    return float(nmi)


def normalized_cross_correlation(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Compute NCC between two images, ignoring zero-valued border pixels
    introduced by warping. Range [-1, 1], higher = better alignment.
    """
    a = _to_gray(img1).ravel()
    b = _to_gray(img2).ravel()

    # Mask: ignore pixels where either image is zero (warp border)
    mask = (a > 0) & (b > 0)
    if mask.sum() < 100:
        # Fallback: use all pixels if masking removes too much
        mask = np.ones_like(mask, dtype=bool)

    a = a[mask]
    b = b[mask]
    a = (a - a.mean()) / (a.std() + 1e-8)
    b = (b - b.mean()) / (b.std() + 1e-8)
    return float(np.mean(a * b))


def alignment_quality(ref: np.ndarray, aligned: np.ndarray) -> dict:
    """Return quality metrics after alignment."""
    return {
        "mutual_information": mutual_information(ref, aligned),
        "ncc": normalized_cross_correlation(ref, aligned),
    }


# ── Phase correlation (original method, translation-only) ────────

def align_by_shift(dapi: np.ndarray, target: np.ndarray):
    """
    Translation-only alignment via phase correlation.
    Returns (aligned, M, (dx, dy)).
    """
    dapi_gray = _to_gray(dapi)
    tgt_gray = _to_gray(target)

    (dx, dy), _ = cv2.phaseCorrelate(tgt_gray, dapi_gray)

    max_shift = min(dapi_gray.shape[:2]) * 0.10
    if abs(dx) > max_shift or abs(dy) > max_shift:
        print("WARNING: estimated shift too large: dx=%.2f, dy=%.2f, use identity transform." % (dx, dy))
        dx, dy = 0.0, 0.0

    M = np.float32([[1, 0, dx], [0, 1, dy]])
    h, w = dapi_gray.shape[:2]
    aligned = _warp(target, M, h, w)
    return aligned, M, (dx, dy)


# ── ORB + RANSAC (rotation + translation + scale) ────────────────

def align_by_orb(
    dapi: np.ndarray,
    target: np.ndarray,
    n_features: int = 5000,
    ratio_thresh: float = 0.75,
    ransac_thresh: float = 5.0,
    min_matches: int = 10,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Feature-based alignment using ORB + RANSAC.
    Handles rotation, translation, and scale differences.

    Returns:
        aligned: warped target image
        M: 2x3 affine matrix (or 3x3 if fallback)
        info: dict with n_matches, inliers, inlier_ratio, quality metrics
    """
    dapi_gray = _to_gray(dapi).astype(np.uint8)
    tgt_gray = _to_gray(target).astype(np.uint8)

    # Normalize to 0-255 for ORB
    def _to_u8(img):
        if img.max() > 255 or img.min() < 0:
            img = ((img - img.min()) / (img.max() - img.min() + 1e-8) * 255)
        return img.astype(np.uint8)

    dapi_u8 = _to_u8(dapi_gray)
    tgt_u8 = _to_u8(tgt_gray)

    # Detect ORB features
    orb = cv2.ORB_create(nfeatures=n_features, scaleFactor=1.2, nlevels=8)
    kp1, desc1 = orb.detectAndCompute(dapi_u8, None)
    kp2, desc2 = orb.detectAndCompute(tgt_u8, None)

    info = {"n_kp_ref": len(kp1), "n_kp_tgt": len(kp2)}

    if desc1 is None or desc2 is None or len(kp1) < min_matches or len(kp2) < min_matches:
        print("WARNING: ORB insufficient keypoints, falling back to phase correlation.")
        aligned, M, shift = align_by_shift(dapi, target)
        info.update({"method": "phase_correlate_fallback", "reason": "insufficient_keypoints"})
        return aligned, M, info

    # Match with KNN + Lowe's ratio test
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    raw_matches = bf.knnMatch(desc2, desc1, k=2)  # query=target, train=dapi

    good = []
    for pair in raw_matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < ratio_thresh * n.distance:
                good.append(m)

    info["n_matches"] = len(good)

    if len(good) < min_matches:
        print("WARNING: ORB too few good matches (%d), falling back to phase correlation." % len(good))
        aligned, M, shift = align_by_shift(dapi, target)
        info.update({"method": "phase_correlate_fallback", "reason": "too_few_matches"})
        return aligned, M, info

    # Extract matched points
    pts_tgt = np.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts_ref = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    # Estimate affine transform (rotation + translation + scale)
    M, inliers = cv2.estimateAffinePartial2D(
        pts_tgt, pts_ref,
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_thresh,
        maxIters=5000,
        confidence=0.99,
    )

    n_inliers = int(inliers.sum()) if inliers is not None else 0
    info["n_inliers"] = n_inliers
    info["inlier_ratio"] = n_inliers / len(good) if good else 0.0

    if M is None or n_inliers < min_matches // 2:
        print("WARNING: ORB RANSAC failed (inliers=%d), falling back to phase correlation." % n_inliers)
        aligned, M, shift = align_by_shift(dapi, target)
        info.update({"method": "phase_correlate_fallback", "reason": "ransac_failed"})
        return aligned, M, info

    h, w = dapi.shape[:2]
    aligned = _warp(target, M, h, w)

    # Quality metrics
    q = alignment_quality(dapi, aligned)
    info.update(q)
    info["method"] = "orb_ransac"
    info["rotation_deg"] = float(np.degrees(np.arctan2(M[1, 0], M[0, 0])))
    info["scale"] = float(np.sqrt(M[0, 0] ** 2 + M[1, 0] ** 2))
    info["tx"] = float(M[0, 2])
    info["ty"] = float(M[1, 2])

    return aligned, M, info


# ── ECC (Enhanced Correlation Coefficient, affine: rotation+shift+scale) ─

def align_by_ecc(
    dapi: np.ndarray,
    target: np.ndarray,
    motion_type: int = cv2.MOTION_AFFINE,
    n_iters: int = 5000,
    eps: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Intensity-based alignment using ECC (Enhanced Correlation Coefficient).
    Handles rotation, translation, and scale. More robust than ORB for
    images with repetitive/symmetric features (e.g. cell nuclei).

    Falls back to phase correlation if ECC fails to converge.
    """
    dapi_gray = _to_gray(dapi)
    tgt_gray = _to_gray(target)

    # ECC needs 8-bit or 32-bit float
    def _to_f32_norm(img):
        g = img.astype(np.float32)
        g -= g.min()
        mx = g.max()
        if mx > 0:
            g /= mx
        return g

    ref_f = _to_f32_norm(dapi_gray)
    tgt_f = _to_f32_norm(tgt_gray)

    # Initial warp matrix
    if motion_type in (cv2.MOTION_AFFINE, cv2.MOTION_EUCLIDEAN):
        warp = np.eye(2, 3, dtype=np.float32)
    elif motion_type == cv2.MOTION_HOMOGRAPHY:
        warp = np.eye(3, 3, dtype=np.float32)
    else:
        warp = np.eye(2, 3, dtype=np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, n_iters, eps)

    try:
        cc, warp_matrix = cv2.findTransformECC(
            ref_f, tgt_f, warp, motion_type,
            criteria=criteria,
            inputMask=None,
            gaussFiltSize=5,
        )
    except cv2.error:
        print("WARNING: ECC failed to converge, falling back to phase correlation.")
        aligned, M, shift = align_by_shift(dapi, target)
        q = alignment_quality(dapi, aligned)
        return aligned, M, {"method": "phase_correlate_fallback", "reason": "ecc_not_converged",
                            "shift": shift, **q}

    h, w = dapi.shape[:2]
    aligned = _warp(target, warp_matrix, h, w)

    q = alignment_quality(dapi, aligned)
    info = {
        "method": "ecc",
        "ecc_coefficient": float(cc),
        "ncc": q["ncc"],
        "mutual_information": q["mutual_information"],
    }

    if motion_type in (cv2.MOTION_AFFINE, cv2.MOTION_EUCLIDEAN):
        info["rotation_deg"] = float(np.degrees(np.arctan2(warp_matrix[1, 0], warp_matrix[0, 0])))
        info["scale"] = float(np.sqrt(warp_matrix[0, 0] ** 2 + warp_matrix[1, 0] ** 2))
        info["tx"] = float(warp_matrix[0, 2])
        info["ty"] = float(warp_matrix[1, 2])

    return aligned, warp_matrix, info


# ── Smart align: tries ORB → ECC → phase correlation ─────────────

def align(
    dapi: np.ndarray,
    target: np.ndarray,
    method: str = "auto",
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Unified alignment interface.

    Args:
        dapi: reference image (DAPI channel)
        target: image to align
        method: "auto" | "orb" | "ecc" | "shift"
        **kwargs: passed to underlying alignment method

    Returns:
        aligned: warped target
        M: transformation matrix
        info: dict with method used, quality metrics, etc.

    Auto fallback chain: ORB → ECC → phase correlation
    """
    if method == "shift":
        aligned, M, shift = align_by_shift(dapi, target)
        q = alignment_quality(dapi, aligned)
        return aligned, M, {"method": "phase_correlate", "shift": shift, **q}

    if method == "orb":
        return align_by_orb(dapi, target, **kwargs)

    if method == "ecc":
        return align_by_ecc(dapi, target, **kwargs)

    # auto: ORB → ECC → phase correlation
    aligned, M, info = align_by_orb(dapi, target, **kwargs)
    if info.get("method", "").endswith("fallback"):
        # ORB failed, try ECC
        print("INFO: ORB failed, trying ECC alignment...")
        aligned, M, info = align_by_ecc(dapi, target, **kwargs)
        if info.get("method", "").endswith("fallback"):
            # ECC also failed, phase correlation was used inside ecc
            q = alignment_quality(dapi, aligned)
            info.update(q)
    return aligned, M, info
