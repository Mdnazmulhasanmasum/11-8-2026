
"""Canonical preprocessing of TinyBrainNet-P21 (identical in training, evaluation and deployment)."""
import numpy as np, cv2
from PIL import Image, ImageOps
DEFAULTS = dict(IMAGE_SIZE=224, CROP_TRIGGER_AREA=0.55, CROP_UNEVEN=0.15, CROP_MARGIN=0.08, INTENSITY_PCTL=99.5, NOISE_SIGMA_THR=0.02)

def load_image_any(src):
    """path or array -> (grey float32 in [0,1], info). Same luma rule for every source."""
    info = {"orig_h": None, "orig_w": None, "error": None}
    try:
        if isinstance(src, np.ndarray):
            a = src.astype(np.float32)
            if a.ndim == 3: a = a[..., :3].mean(-1)
            mx = 65535.0 if src.dtype == np.uint16 else (255.0 if src.dtype == np.uint8 else max(float(a.max()), 1e-6)); g = a / mx
        else:
            im = ImageOps.exif_transpose(Image.open(src))
            if im.mode in ("I;16", "I;16B", "I;16L", "I"):
                a = np.array(im, dtype=np.float32); g = a / (65535.0 if a.max() > 255 else 255.0)
            else:
                if im.mode in ("RGBA", "LA", "P"):
                    im = im.convert("RGBA"); bg = Image.new("RGBA", im.size, (0, 0, 0, 255)); im = Image.alpha_composite(bg, im)
                g = np.array(im.convert("L"), dtype=np.float32) / 255.0
    except Exception as e:
        info["error"] = f"{type(e).__name__}: {e}"; return None, info
    if g.ndim != 2 or min(g.shape) < 8:
        info["error"] = f"unexpected image shape {g.shape}"; return None, info
    info["orig_h"], info["orig_w"] = g.shape
    return np.clip(g, 0, 1).astype(np.float32), info

def _trim_uniform_border(g8, std_thr=3.0, max_frac=0.25):
    H, W = g8.shape; t, b, l, r = 0, H, 0, W
    while t < H * max_frac and g8[t].std() < std_thr and abs(float(g8[t].mean()) - float(g8[0].mean())) < 20: t += 1
    while H - b < H * max_frac and g8[b - 1].std() < std_thr and abs(float(g8[b - 1].mean()) - float(g8[-1].mean())) < 20: b -= 1
    while l < W * max_frac and g8[:, l].std() < std_thr and abs(float(g8[:, l].mean()) - float(g8[:, 0].mean())) < 20: l += 1
    while W - r < W * max_frac and g8[:, r - 1].std() < std_thr and abs(float(g8[:, r - 1].mean()) - float(g8[:, -1].mean())) < 20: r -= 1
    return t, b, l, r

def head_box(g):
    """-> (box (y0,y1,x0,x1) or None, fallback_step). 1: Otsu components >= 1 % of the image; 2: foreground-threshold box; 3: none."""
    g8 = (g * 255).astype(np.uint8)
    if float(np.percentile(g8, 99) - np.percentile(g8, 1)) < 12: return None, 3          # almost flat image: no reliable head
    t, b, l, r = _trim_uniform_border(g8); roi = g8[t:b, l:r]
    blur = cv2.GaussianBlur(roi, (5, 5), 0); otsu, _ = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    fg = cv2.morphologyEx((blur > max(8.0, 0.5 * otsu)).astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, _, st, _ = cv2.connectedComponentsWithStats(fg, 8)
    keep = [i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] >= 0.01 * roi.size]
    if keep:
        y0 = min(st[i, cv2.CC_STAT_TOP] for i in keep); y1 = max(st[i, cv2.CC_STAT_TOP] + st[i, cv2.CC_STAT_HEIGHT] for i in keep)
        x0 = min(st[i, cv2.CC_STAT_LEFT] for i in keep); x1 = max(st[i, cv2.CC_STAT_LEFT] + st[i, cv2.CC_STAT_WIDTH] for i in keep)
        return (t + y0, t + y1, l + x0, l + x1), 1
    ys, xs = np.where(cv2.GaussianBlur(g8, (5, 5), 0) > 0.05 * max(int(g8.max()), 1))
    if len(ys) >= 0.005 * g8.size: return (int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1), 2
    return None, 3

def canonical_geometry(g, P=DEFAULTS, crop=True):
    H, W = g.shape; box, step = head_box(g)
    p = {"H": H, "W": W, "crop": (0, H, 0, W), "crop_applied": False, "fallback_step": step, "bbox": box}
    if crop and box is not None:
        y0, y1, x0, x1 = box; bh, bw = y1 - y0, x1 - x0; m = (y0 / H, (H - y1) / H, x0 / W, (W - x1) / W)
        uneven = max(abs(m[0] - m[1]), abs(m[2] - m[3])) > P["CROP_UNEVEN"]
        if bh * bw / (H * W) < P["CROP_TRIGGER_AREA"] or uneven:
            k = int(round(P["CROP_MARGIN"] * max(bh, bw)))
            p["crop"] = (max(0, y0 - k), min(H, y1 + k), max(0, x0 - k), min(W, x1 + k)); p["crop_applied"] = True
    y0, y1, x0, x1 = p["crop"]; h, w = y1 - y0, x1 - x0; s = max(h, w)
    p["pad"] = ((s - h) // 2, s - h - (s - h) // 2, (s - w) // 2, s - w - (s - w) // 2)
    return p

def apply_geometry(a, p, kind="image", size=224):
    """kind: image (area / cubic), mask (nearest), prob (linear). The same crop + pad + resize for all three."""
    y0, y1, x0, x1 = p["crop"]; a = cv2.copyMakeBorder(np.ascontiguousarray(a[y0:y1, x0:x1]), *p["pad"], cv2.BORDER_CONSTANT, value=0)
    interp = {"mask": cv2.INTER_NEAREST, "prob": cv2.INTER_LINEAR}.get(kind, cv2.INTER_AREA if a.shape[0] > size else cv2.INTER_CUBIC)
    return cv2.resize(a, (size, size), interpolation=interp)

def canonical_preprocess(src, mask=None, P=DEFAULTS, crop=True):
    """-> (img float32 [0,1] 224x224, mask uint8 {0,1} or None, info)."""
    g, info = load_image_any(src)
    if g is None: return None, None, info
    p = canonical_geometry(g, P, crop); S = P["IMAGE_SIZE"]
    img = apply_geometry(g, p, "image", S); head = img > 0.04
    scale = float(np.percentile(img[head], P["INTENSITY_PCTL"])) if head.sum() > 50 else 1.0
    img = np.clip(img / max(scale, 0.05), 0, 1).astype(np.float32); m = None
    if mask is not None:
        if mask.shape != g.shape: mask = cv2.resize(mask, (g.shape[1], g.shape[0]), interpolation=cv2.INTER_NEAREST)
        m = (apply_geometry(mask, p, "mask", S) > 0).astype(np.uint8)
    info.update({k: p[k] for k in ("crop", "pad", "crop_applied", "fallback_step", "H", "W")}, intensity_scale=scale)
    return img, m, info

def to_cache_u8(img): return np.clip(np.round(img * 255.0), 0, 255).astype(np.uint8)

_IMMERKAER = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
def estimate_noise_sigma(img):
    img = np.ascontiguousarray(img, dtype=np.float32); r = np.abs(cv2.filter2D(img, -1, _IMMERKAER))[1:-1, 1:-1]
    gm = cv2.magnitude(cv2.Sobel(img, cv2.CV_32F, 1, 0), cv2.Sobel(img, cv2.CV_32F, 0, 1))[1:-1, 1:-1]; tissue = img[1:-1, 1:-1] > 0.08
    if tissue.sum() < 200: return 0.0
    flat = tissue & (gm <= np.percentile(gm[tissue], 70)); return float(np.sqrt(np.pi / 2) / 6 * r[flat].mean())

def finalize(img, P=DEFAULTS, adaptive_filter=True):
    """Last step for every image (training after augmentation, evaluation, deployment): noise-adaptive median filter."""
    img = np.ascontiguousarray(img, dtype=np.float32); s = estimate_noise_sigma(img)
    if adaptive_filter and s > P["NOISE_SIGMA_THR"]:
        k = 3 if s < 0.05 else 5; return cv2.medianBlur(to_cache_u8(img), k).astype(np.float32) / 255.0, s, k
    return img, s, 0

def model_input_from_u8(u8, mean, std, P=DEFAULTS, adaptive_filter=True):
    """uint8 224x224 canonical image -> float32 [1,224,224] network input."""
    x, _, _ = finalize(u8.astype(np.float32) / 255.0, P, adaptive_filter)
    return ((x - mean) / std).astype(np.float32)[None]

def postprocess_mask(prob_u8, min_cc=0):
    """Binary mask at probability 0.5, then remove 4-connected components smaller than min_cc pixels (min_cc set on validation)."""
    m = prob_u8 > 127
    if min_cc <= 0 or not m.any(): return m
    n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=4)
    keep = np.zeros(n, bool); keep[1:] = st[1:, cv2.CC_STAT_AREA] >= min_cc
    return keep[lab]

def preprocess_for_model(src, mean, std, P=DEFAULTS, adaptive_filter=True):
    """Deployment entry point: any image path / array -> (network input [1,224,224], canonical uint8 image, info)."""
    img, _, info = canonical_preprocess(src, None, P)
    if img is None: return None, None, info
    u8 = to_cache_u8(img); return model_input_from_u8(u8, mean, std, P, adaptive_filter), u8, info
