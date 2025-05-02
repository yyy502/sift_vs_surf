import cv2
import numpy as np
from typing import List, Tuple, Dict

def _closest_keypoints(user_pts: List[Tuple[int, int]], keypoints: List[cv2.KeyPoint],
                       radius: float = 5.0) -> Dict[int, int]:
    """
    Map each user‑supplied (x, y) to the index of the nearest detected key‑point
    within `radius` pixels.  Returns a dict {user_idx -> keypoint_idx}.
    """
    kp_coords = np.array([[kp.pt[0], kp.pt[1]] for kp in keypoints])
    mapping = {}
    for i, (ux, uy) in enumerate(user_pts):
        dists = np.linalg.norm(kp_coords - np.array([ux, uy]), axis=1)
        nearest = np.argmin(dists)
        if dists[nearest] <= radius:
            mapping[i] = int(nearest)
    return mapping

def _draw_matches_full(imgA, imgB, kpA, kpB, matches, max_matches=100, title="matches"):
    """Visualise matches using matplotlib."""
    import matplotlib.pyplot as plt

    drawn = cv2.drawMatches(
        imgA, kpA,
        imgB, kpB,
        matches[:max_matches], None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    plt.figure(figsize=(12, 6))
    plt.imshow(cv2.cvtColor(drawn, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title(title)
    plt.show()

def _draw_matches_manual(imgA, imgB, kpA, kpB, matches, max_matches=100, title="matches"):
    """Visualise matches using matplotlib."""
    import matplotlib.pyplot as plt

    drawn = cv2.drawMatches(
        imgA, kpA,
        imgB, kpB,
        matches[:max_matches], None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    # 2) Then over‑draw each match with your desired thickness
    hA, wA = imgA.shape[:2]
    for m in matches[:max_matches]:
        # note: match.trainIdx comes from the second image, so add wA to x
        ptA = (int(kpA[m.queryIdx].pt[0]), int(kpA[m.queryIdx].pt[1]))
        ptB = (int(kpB[m.trainIdx].pt[0] + wA), int(kpB[m.trainIdx].pt[1]))
        # sample original cv2.drawMatches line color at midpoint
        mid_x = (ptA[0] + ptB[0]) // 2
        mid_y = (ptA[1] + ptB[1]) // 2
        b, g, r = drawn[mid_y, mid_x]
        cv2.line(drawn, ptA, ptB, color=(int(b), int(g), int(r)), thickness=3)

    plt.figure(figsize=(12, 6))
    plt.imshow(cv2.cvtColor(drawn, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title(title)
    plt.show()


def run_sift_surf_and_filter(
    imgA_path: str,
    imgB_path: str,
    user_points: List[Tuple[int, int]],
    radius_px: float = 5.0,
    max_ret_matches: int = 300,
):
    """
    Compare SIFT and SURF descriptors given manual interest points.

    Parameters
    ----------
    imgA_path : str
        Path to the *source* image (where manual points are defined).
    imgB_path : str
        Path to the *destination* image.
    user_points : list[tuple[int,int]]
        Manual (x, y) coordinates in image A.
    radius_px : float, optional
        Max pixel distance for snapping a user point to the nearest detected key‑point.
    max_ret_matches : int, optional
        Number of matches to keep for visualisation.

    Returns
    -------
    results : dict
        {
            'sift_full': (kpA_sift, kpB_sift, matches_sift),
            'surf_full': (kpA_surf, kpB_surf, matches_surf),
            'sift_user': (kpA_sift, kpB_sift, matches_sift_user),
            'surf_user': (kpA_surf, kpB_surf, matches_surf_user),
        }
        Key‑points lists are exactly those produced by OpenCV detectAndCompute.
        The *_user variants contain only matches whose queryIdx comes from a
        key‑point snapped to the manual interest points.
    """
    # ----- load images (as colour for display, grayscale for ops) -----
    imgA_color = cv2.imread(imgA_path, cv2.IMREAD_COLOR)
    imgB_color = cv2.imread(imgB_path, cv2.IMREAD_COLOR)
    imgA_gray = cv2.cvtColor(imgA_color, cv2.COLOR_BGR2GRAY)
    imgB_gray = cv2.cvtColor(imgB_color, cv2.COLOR_BGR2GRAY)

    # ----- build extractors -----
    sift = cv2.SIFT_create()
    surf = cv2.xfeatures2d.SURF_create(extended=False)  # set extended=True for 128‑D

    # ----- detect & compute -----
    kpA_sift, descA_sift = sift.detectAndCompute(imgA_gray, None)
    kpB_sift, descB_sift = sift.detectAndCompute(imgB_gray, None)

    kpA_surf, descA_surf = surf.detectAndCompute(imgA_gray, None)
    kpB_surf, descB_surf = surf.detectAndCompute(imgB_gray, None)


    # ----- snap manual points -----
    sift_map = _closest_keypoints(user_points, kpA_sift, radius_px)
    surf_map = _closest_keypoints(user_points, kpA_surf, radius_px)

    # compute matches
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches_sift = sorted(
        bf.match(descA_sift, descB_sift), key=lambda m: m.distance
    )#[:max_ret_matches]

    matches_surf = sorted(
        bf.match(descA_surf, descB_surf), key=lambda m: m.distance
    )#[:max_ret_matches]

    # filter matches whose queryIdx is in snapped set
    sift_user_matches = [
        m for m in matches_sift if m.queryIdx in sift_map.values()
    ]
    surf_user_matches = [
        m for m in matches_surf if m.queryIdx in surf_map.values()
    ]
    
    breakpoint()

    # package results
    results = {
        "sift_full": (kpA_sift, kpB_sift, matches_sift),
        "surf_full": (kpA_surf, kpB_surf, matches_surf),
        "sift_user": (kpA_sift, kpB_sift, sift_user_matches),
        "surf_user": (kpA_surf, kpB_surf, surf_user_matches),
    }

    # ----- visualise -----
    print("Full SIFT matches")
    _draw_matches_full(imgA_color, imgB_color, *results["sift_full"], title="SIFT – full")

    print("Full SURF matches")
    _draw_matches_full(imgA_color, imgB_color, *results["surf_full"], title="SURF – full")

    print("SIFT matches on manual points")
    _draw_matches_manual(imgA_color, imgB_color, *results["sift_user"], title="SIFT – manual")

    print("SURF matches on manual points")
    _draw_matches_manual(imgA_color, imgB_color, *results["surf_user"], title="SURF – manual")

    return results

import matplotlib.pyplot as plt
img = plt.imread("/Users/yuyanyang/Downloads/cat_image1.JPG")
plt.imshow(img); pts = plt.ginput(10)
print(pts)

results = run_sift_surf_and_filter(
    imgA_path="/Users/yuyanyang/Downloads/cat_image1.JPG",
    imgB_path="/Users/yuyanyang/Downloads/cat_image2.JPG",
    user_points=pts,   # your ear clicks
    radius_px=50                                      # snap tolerance
)