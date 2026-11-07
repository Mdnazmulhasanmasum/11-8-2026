# TinyBrainNet-P21 deployment bundle 
Research prototype. Not a medical device. Requires independent clinical validation before any clinical use.
Model confidence is not clinical certainty.

## Use
```python
from infer import TinyBrainNetPredictor
p = TinyBrainNetPredictor(".")            # CPU
r = p.predict("slice.jpg")                # status ok / review / failed, label, probabilities, mask_224, cam_14, reasons
```
Command line: `python infer.py image1.jpg image2.png`

## Files
model.py (network), preprocess.py (the exact training preprocessing), infer.py, part21_best_weights.pt (EMA best, SHA-256 in meta.json),
meta.json, config.json, calibration.json, ood_stats.npz, best_model.onnx (verified parity), model_torchscript.pt (verified parity).

## What it was selected on
Arm E2, chosen by the pre-declared validation rule: E2 if S_BRISC_clean_val(E2) >= S_BRISC_clean_val(E1) - 0.01, else E1 (pre-declared). No test result was used for selection.
Evaluation tiers and their overlap labels: see reports/final_evaluation.json and reports/comparison_table.csv.
