# Model card: TinyBrainNet-P21
- Architecture: Part-16 UNI-NET skeleton, 1-channel stem, mask-supervised attention head (cam), auxiliary 56x56 head; 496,503 trainable parameters (limit 500,000).
- Initialisation: plan B (v6 classifier checkpoint); classification teachers ['v6']; segmentation teacher tinyunet_pre.
- Training data: arm E2 (BRISC2025 + Nickparvar Training (non-duplicate) + Br35H 70% (binary labels)).
- Inputs: one 2D grey MRI slice; preprocessing in preprocess.py (identical to training).
- Outputs: 4-class probabilities (temperature 1.064, fitted on validation), tumour mask, learned attention map, review status.
- Evaluation: six tiers with explicit overlap labels; only TCGA-LGG is independent of BRISC and it is glioma-only with FLAIR masks.
- Limitations: no patient IDs in BRISC / Nickparvar / Br35H (near-duplicate groups used instead); single sequence (T1 contrast-enhanced);
  external generalisation remains limited; not validated prospectively or clinically.
- Not for clinical use. Research prototype.
