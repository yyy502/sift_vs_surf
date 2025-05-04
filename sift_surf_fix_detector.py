# ---------- SURF‑detected key‑points + SIFT vs SURF descriptors -------------
#
# Detect interest points **once** with the SURF Fast‑Hessian detector, then
# compute TWO different descriptors on those *same* points:
#   • the classic 128‑D SIFT gradient‑histogram descriptor
#   • the standard 64‑D SURF Haar‑wavelet descriptor
#
# This lets you compare descriptor quality without confounding detector choice.
#
# ---------------------------------------------------------------------------
import cv2
import numpy as np
from typing import List, Tuple, Dict


# ---------- small utilities -------------------------------------------------
def _closest_keypoints(user_pts: List[Tuple[int, int]], keypoints: List[cv2.KeyPoint],
                       radius: float = 5.0) -> Dict[int, int]:
    """Snap each manual (x, y) to the nearest detected key‑point within *radius* pixels."""
    kp_xy = np.array([kp.pt for kp in keypoints])     # (N,2)
    mapping = {}
    for i, (ux, uy) in enumerate(user_pts):
        d = np.linalg.norm(kp_xy - (ux, uy), axis=1)
        j = np.argmin(d)
        if d[j] <= radius:
            mapping[i] = int(j)
    return mapping


def _draw_matches(imgA, imgB, kpA, kpB, matches, title, thick=2, max_draw=100):
    """Visualise the first *max_draw* matches (thicker coloured lines)."""
    import matplotlib.pyplot as plt

    vis = cv2.drawMatches(imgA, kpA, imgB, kpB,
                          matches[:max_draw], None,
                          flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # over‑draw thicker match lines for clarity
    wA = imgA.shape[1]
    for m in matches[:max_draw]:
        a = tuple(map(int, kpA[m.queryIdx].pt))
        b = (int(kpB[m.trainIdx].pt[0] + wA), int(kpB[m.trainIdx].pt[1]))
        mid = ((a[0] + b[0]) // 2, (a[1] + b[1]) // 2)
        colour = vis[mid[1], mid[0]].tolist()         # sample original colour
        cv2.line(vis, a, b, colour, thick)

    plt.figure(figsize=(12, 6))
    plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title(title)
    plt.show()


# ---------- main pipeline ---------------------------------------------------
def compare_descriptors_on_surf(
    imgA_path: str,
    imgB_path: str,
    user_points: List[Tuple[int, int]],
    snap_radius_px: float = 5.0,
    max_visual_matches: int = 300,
):
    """
    1. Detect key‑points in both images with the SURF Fast‑Hessian detector.
    2. Compute two descriptor sets **on the same key‑points**:
       SIFT (128‑D) and SURF (64‑D).
    3. BF‑match each descriptor set and optionally filter by user‑supplied points.
    """
    # -- load & grey --
    imgA = cv2.imread(imgA_path, cv2.IMREAD_COLOR)
    imgB = cv2.imread(imgB_path, cv2.IMREAD_COLOR)
    gA = cv2.cvtColor(imgA, cv2.COLOR_BGR2GRAY)
    gB = cv2.cvtColor(imgB, cv2.COLOR_BGR2GRAY)

    # -- 1) SURF detector only (no descriptor) --
    surf_det = cv2.xfeatures2d.SURF_create(hessianThreshold=400)  # detector
    kpA = surf_det.detect(gA, None)
    kpB = surf_det.detect(gB, None)

    # -- 2) descriptors on the SURF key‑points --
    sift = cv2.SIFT_create()
    surf_desc = cv2.xfeatures2d.SURF_create(extended=False)       # 64‑D

    _, descA_sift = sift.compute(gA, kpA)
    _, descB_sift = sift.compute(gB, kpB)

    _, descA_surf = surf_desc.compute(gA, kpA)
    _, descB_surf = surf_desc.compute(gB, kpB)

    # -- 3) matchers --
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches_sift = sorted(bf.match(descA_sift, descB_sift), key=lambda m: m.distance)
    matches_surf = sorted(bf.match(descA_surf, descB_surf), key=lambda m: m.distance)

    # -- 4) snap manual points and filter matches --
    surf_map = _closest_keypoints(user_points, kpA, snap_radius_px)
    sift_user = [m for m in matches_sift if m.queryIdx in surf_map.values()]
    surf_user = [m for m in matches_surf if m.queryIdx in surf_map.values()]

    # -- 5) visualise --
    print("SIFT‑descriptor matches (SURF detector)")
    _draw_matches(imgA, imgB, kpA, kpB, matches_sift,
                  "Homography transformation: SIFT descriptor (on surf detector) – all", max_draw=max_visual_matches)
    print("SURF‑descriptor matches (SURF detector)")
    _draw_matches(imgA, imgB, kpA, kpB, matches_surf,
                  "Homography transformation: SURF descriptor (on surf detector) – all", max_draw=max_visual_matches)

    print("SIFT matches on manual points")
    _draw_matches(imgA, imgB, kpA, kpB, sift_user,
                  "Homography transformation: SIFT descriptor – manual", thick=3)
    print("SURF matches on manual points")
    _draw_matches(imgA, imgB, kpA, kpB, surf_user,
                  "Homography transformation: SURF descriptor – manual", thick=3)

    return {
        "sift_full":  (kpA, kpB, matches_sift),
        "surf_full":  (kpA, kpB, matches_surf),
        "sift_user":  (kpA, kpB, sift_user),
        "surf_user":  (kpA, kpB, surf_user),
    }


# ---------- example use -----------------------------------------------------
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    imgA_path = "./images/cat_image1.JPG"
    imgB_path = "./images/cat_image2.JPG"

    # manually click points on imgA
    imgA_disp = plt.imread(imgA_path)
    plt.imshow(imgA_disp); plt.title("Click up to 10 interest points"); pts = plt.ginput(10, timeout=0)
    plt.close()
    print("Manual points:", pts)

    compare_descriptors_on_surf(
        imgA_path, imgB_path,
        user_points=pts,
        snap_radius_px=50,
        max_visual_matches=300,
    )