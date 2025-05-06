# Experiment Overview
We set out to compare the matching performance of SIFT and SURF descriptors on two views of the same scene. First, we ran the full SIFT and SURF pipelines on both images and visualized all automatically detected correspondences—however, those “global” match plots were overwhelmed by hundreds of lines, making it impossible to assess match quality at specific locations.

# Feature Descriptor Analysis

## Descriptor Comparison Framework

SIFT vs. SURF evaluation using Oxford VGG Affine Covariant Regions dataset
Matching methodology based on region overlap and Euclidean distance threshold

## Distortion Resilience Testing

Evaluation under five image distortion categories (https://www.robots.ox.ac.uk/~vgg/research/affine/index.html)

Measurement of recall, precision, and runtime metrics

# Classification Methods
Evaluation for Cat vs Dog Binary Classification (https://www.kaggle.com/datasets/andrewmvd/animal-faces)

## Benchmark: Bag of Visual Words (BoVW)

Implementation using SIFT descriptors

K-means clustering (100 visual words)
Visual word frequency histograms
SVM classification

## Advanced Method: Spatial Pyramid Matching (SPM)

Dense SIFT implementation on regular grid
Multi-level spatial pyramids

Level 0: 1×1 grid

Level 1: 2×2 grid

Level 2: 4×4 grid

Level 3: 8×8 grid


SVM classification with same framework as benchmark

#  Comparative Performance Analysis
## Parameter Optimization

Vocabulary size variations (M=16, M=50, M=100)
Pyramid level effectiveness
Configuration optimization

## Benchmark Comparison

Quantitative comparison between BoVW and SPM
Analysis of spatial information contribution to performance


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
