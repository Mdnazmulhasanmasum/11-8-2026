
"""TinyBrainNet-P21 inference (CPU by default). RESEARCH PROTOTYPE: not a medical device, not clinically validated.
Model confidence is not clinical certainty; every output must be reviewed by a qualified clinician."""
import json, numpy as np, torch, cv2
from pathlib import Path
from model import build_model

def image_chroma(src):
    """mean |max(RGB) - min(RGB)| / 255; ~0 for a greyscale MRI, large for colour photos / screenshots"""
    a = src if isinstance(src, np.ndarray) else cv2.imread(str(src), cv2.IMREAD_COLOR)
    if a is None or a.ndim < 3 or a.shape[2] < 3: return 0.0
    a = a[..., :3].astype(np.float32); return float((a.max(2) - a.min(2)).mean() / 255.0)
from preprocess import preprocess_for_model, postprocess_mask, to_cache_u8

class TinyBrainNetPredictor:
    def __init__(self, bundle_dir=".", device="cpu"):
        d = Path(bundle_dir); self.meta = json.loads((d / "meta.json").read_text()); self.device = torch.device(device)
        self.model = build_model(self.meta["model"]["use_adapter"])
        self.model.load_state_dict(torch.load(d / self.meta["model"]["weights_file"], map_location="cpu", weights_only=True), strict=True)
        self.model.to(self.device).eval(); o = np.load(d / "ood_stats.npz"); self.mu, self.prec, self.ood_thr = o["mu"], o["prec"], float(o["thr"])
        p = self.meta["preprocessing"]; self.P, self.mean, self.std = p["params"], p["mean"], p["std"]; self.cal = self.meta["calibration"]
        self.classes = self.meta["classes"]

    @torch.no_grad()
    def _run(self, x):
        o = self.model(torch.from_numpy(x[None]).to(self.device), "both")
        return (o["cls_logits"][0].float().cpu().numpy(), torch.sigmoid(o["seg_logits"][0, 0].float()).cpu().numpy(),
                o["cam"][0, 0].float().cpu().numpy(), o["feat"][0].float().cpu().numpy())

    def predict(self, src):
        """src: image path or numpy array. Returns a dict; 'status' is 'ok', 'review' or 'failed'."""
        chroma = image_chroma(src)
        x, u8, info = preprocess_for_model(src, self.mean, self.std, self.P)
        if x is None: return {"status": "failed", "reasons": [f"could not read / preprocess the image: {info.get('error')}"]}
        logits, prob, cam, feat = self._run(x); z = logits / self.cal["temperature"]; pz = np.exp(z - z.max()); pz /= pz.sum()
        k = int(pz.argmax()); mask = postprocess_mask(to_cache_u8(prob), self.cal["seg_min_component_px"]); area = int(mask.sum())
        dvec = feat.astype(np.float64) - self.mu; ood = float(np.sqrt(dvec @ self.prec @ dvec)); reasons = []
        thr = self.cal.get("review_threshold")
        if chroma > self.cal.get("colour_reject", 1.0): reasons.append("colour image: not a greyscale MRI slice")
        if thr is None or pz[k] < thr: reasons.append("confidence below the validation threshold that proves >= 95 % accuracy on accepted images")
        if ood > self.ood_thr: reasons.append("image is unlike the training data (Mahalanobis distance above the validation 99th percentile)")
        if info.get("fallback_step", 1) >= 2: reasons.append(f"head detection used fallback step {info.get('fallback_step')}")
        seg_tumour = area > self.cal["seg_area_tau_px"]
        if (self.classes[k] == "No_tumor") == seg_tumour: reasons.append("classification and segmentation disagree")
        return {"status": "review" if reasons else "ok", "reasons": reasons, "label": self.classes[k], "confidence": float(pz[k]),
                "probabilities": {c: float(p) for c, p in zip(self.classes, pz)}, "logits": logits.tolist(), "tumour_area_px_224": area,
                "mask_224": mask.astype(np.uint8), "cam_14": cam, "canonical_image_224": u8, "ood_distance": ood, "colour_chroma": chroma,
                "preprocessing": {k_: info.get(k_) for k_ in ("crop_applied", "fallback_step", "orig_h", "orig_w")},
                "disclaimer": "Research prototype; requires independent clinical validation. Confidence is not clinical certainty."}

    def classify(self, src):
        r = self.predict(src); return {k: r.get(k) for k in ("status", "reasons", "label", "confidence", "probabilities")}

    def segment(self, src):
        r = self.predict(src); return {k: r.get(k) for k in ("status", "reasons", "mask_224", "tumour_area_px_224", "canonical_image_224")}

if __name__ == "__main__":
    import sys
    p = TinyBrainNetPredictor(Path(__file__).parent)
    for s in sys.argv[1:]:
        r = p.predict(s); print(s, r["status"], r.get("label"), round(r.get("confidence", 0), 4), r.get("tumour_area_px_224"), r["reasons"])
