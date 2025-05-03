# Experiment Overview
We set out to compare the matching performance of SIFT and SURF descriptors on two views of the same scene. First, we ran the full SIFT and SURF pipelines on both images and visualized all automatically detected correspondences—however, those “global” match plots were overwhelmed by hundreds of lines, making it impossible to assess match quality at specific locations.

# Another way
1. Feature Extraction: Extract SIFT and SURF features from animal images
2. Distortion Analysis: Apply various distortions to images and compare descriptor stability
3. Descriptor Evaluation: Measure and compare recall and precision metrics
4. Spatial Pyramid Integration: Apply Spatial Pyramid Matching to add spatial information
5. Classification Performance: Evaluate categorization accuracy with different combinations

---

# Manual‑Point Refinement
1. **Landmark Selection**  
   We manually clicked three distinctive points in the source image (e.g. the cat’s left ear tip, right ear tip, and nose bridge).  
2. **Key‑Point Snapping**  
   For each click, we searched within a 50 px radius for the nearest automatically detected SIFT or SURF key‑point.  
   - Only clicked points that had a detected key‑point within that radius were considered further.  
3. **Descriptor Extraction & Matching**  
   We computed SIFT and SURF descriptors at those snapped key‑points in both images, then ran cross‑checked Brute‑Force matching.  
4. **Filtered Visualization**  
   We plotted only the matches originating from our three landmarks—each method yields at most three lines—so we can directly judge which descriptor produced reliable correspondences at those exact locations.

---

# Rationale & Benefits
- **Contrast Global vs. Local**  
  Showing both the noisy full-image matches and the focused manual-point matches highlights overall behavior versus point‑specific performance.  
- **True Algorithm Outputs**  
  By snapping only to actually detected key‑points (no manual injection), we evaluate each descriptor’s natural ability to find and match features at our points of interest.  
- **Clean, Interpretable Results**  
  The pared‑down plots make it trivial to see which landmarks were successfully matched and with what confidence, yielding an immediate, qualitative comparison of SIFT versus SURF.  
