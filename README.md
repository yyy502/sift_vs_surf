# Experiment Overview
We set out to compare the matching performance of SIFT and SURF descriptors on two views of the same scene. First, we ran the full SIFT and SURF pipelines on both images and visualized all automatically detected correspondences—however, those “global” match plots were overwhelmed by hundreds of lines, making it impossible to assess match quality at specific locations.

# Another Methodology
1. Feature Extraction

Implementation of SIFT (Scale-Invariant Feature Transform) and SURF (Speeded-Up Robust Features) descriptors
Dataset: Oxford VGG Affine Covariant Regions dataset (https://www.robots.ox.ac.uk/~vgg/research/affine/index.html)
Comparative extraction and analysis of keypoints and descriptors across image sets

2. Distortion Analysis

Application of various image transformations:
Geometric: rotation, scaling
Photometric: illumination changes, blur, compression, viewpoint change


Quantitative measurement of descriptor stability and invariance properties
Visual and statistical analysis of feature resilience under transformations

3. Descriptor Evaluation

Implementation of precision and recall metrics for feature matching
ROC curve analysis for descriptor performance
Cross-comparison between SIFT and SURF under different parameters
Identification of optimal descriptor configurations

4. SIFT-Only Classification

Application to Animal Faces dataset (https://www.kaggle.com/datasets/andrewmvd/animal-faces)
Implementation of Bag of Visual Words (BoVW) representation
Feature extraction, vocabulary creation, and histogram generation
SVM classification with cross-validation
Performance evaluation using confusion matrices and accuracy metrics

5. Spatial Pyramid Integration

Implementation of Spatial Pyramid Matching framework
Multi-level spatial feature aggregation (1×1, 2×2, 4×4, 8×8 grids)
Weighted histogram combination across pyramid levels
Vocabulary size optimization (M=16, M=50, M=100)
Feature vector normalization techniques

6. Classification Performance Comparison

Systematic evaluation of SIFT-Only vs. SIFT+SPM approaches
Analysis of accuracy improvements with spatial information
Influence of vocabulary size and pyramid levels on performance
Visualization of classification results using confusion matrices
Statistical significance testing of performance differences

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
