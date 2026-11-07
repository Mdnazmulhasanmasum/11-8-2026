"""TinyBrainNet - Streamlit app (research prototype). Run:  streamlit run app.py
Everything the model does is in bundle/ (the exact code and weights verified in the notebook)."""
import io, json, sys, tempfile, zipfile
from pathlib import Path
import numpy as np
import cv2
from PIL import Image

ROOT = Path(__file__).resolve().parent
BUNDLE = ROOT / "bundle"
sys.path.insert(0, str(BUNDLE))
from infer import TinyBrainNetPredictor          # noqa: E402  (bundle code, unchanged)
from report import make_pdf                      # noqa: E402

EXT = ("png", "jpg", "jpeg", "bmp", "tif", "tiff")


# ------------------------------------------------------------------ analysis (no Streamlit here: importable and testable)
def load_predictor():
    return TinyBrainNetPredictor(BUNDLE, device="cpu")

def load_card():
    p = ROOT / "model_card.json"
    return json.loads(p.read_text()) if p.exists() else {}

def analyse(pred, image_bytes, name="image.png"):
    """Runs the deployed predictor on the uploaded bytes through a temporary file (same path as the notebook tests)."""
    suf = Path(name).suffix.lower() or ".png"
    with tempfile.NamedTemporaryFile(suffix=suf, delete=False) as f:
        f.write(image_bytes); tmp = f.name
    try:
        r = pred.predict(tmp)
    finally:
        Path(tmp).unlink(missing_ok=True)
    return r

def original_rgb(image_bytes):
    a = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
    if a is None:
        return np.array(Image.open(io.BytesIO(image_bytes)).convert("RGB"))
    if a.dtype != np.uint8:
        a = cv2.normalize(a.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if a.ndim == 2: return cv2.cvtColor(a, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(a[..., :3], cv2.COLOR_BGR2RGB)

def panels(r, mask_alpha=0.45, show_mask=True, cam_alpha=0.45):
    """RGB panels: model input, tumour outline/fill (only if a tumour is predicted), attention map."""
    u8 = r["canonical_image_224"]; base = cv2.cvtColor(u8, cv2.COLOR_GRAY2RGB)
    seg = base.copy(); m = r["mask_224"].astype(bool)
    if show_mask and m.any() and r.get("label") != "No_tumor":
        fill = seg.copy(); fill[m] = (239, 68, 68); seg = cv2.addWeighted(fill, mask_alpha, seg, 1 - mask_alpha, 0)
        cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); cv2.drawContours(seg, cs, -1, (250, 204, 21), 1)
    cam = cv2.resize(r["cam_14"].astype(np.float32), (224, 224), interpolation=cv2.INTER_CUBIC)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    heat = cv2.cvtColor(cv2.applyColorMap((255 * cam).astype(np.uint8), cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)
    att = cv2.addWeighted(heat, cam_alpha, base, 1 - cam_alpha, 0)
    seg_title = "Tumour segmentation" if (m.any() and r.get("label") != "No_tumor") else "Segmentation (no tumour predicted - nothing drawn)"
    return {"Model input (canonical 224 x 224)": base, seg_title: seg, "Attention map (where the classifier pooled evidence)": att}

def head_fraction(r):
    head = r["canonical_image_224"] > 15
    return float(min(1.0, (r["mask_224"].astype(bool) & head).sum() / max(1, head.sum())))

def report_pdf(name, image_bytes, r, pred, card):
    return make_pdf(name, image_bytes, r, panels(r) if r.get("status") != "failed" else {}, pred.meta, card)

def summary_row(name, r):
    return {"file": name, "status": r.get("status"), "prediction": r.get("label") if r.get("status") == "ok" else "",
            "best guess (if referred)": r.get("label") if r.get("status") == "review" else "",
            "confidence %": round(100 * r.get("confidence", 0.0), 1), "tumour area px": r.get("tumour_area_px_224", 0),
            "OOD distance": round(r.get("ood_distance", 0.0), 2), "reasons": "; ".join(r.get("reasons", []))}


# ------------------------------------------------------------------ user interface
CSS = """
<style>
.block-container {padding-top: 1.6rem; max-width: 1250px;}
.tb-hero {background: linear-gradient(120deg,#0e7490 0%,#155e75 60%,#164e63 100%); color:#fff; padding:18px 24px; border-radius:14px; margin-bottom:14px;}
.tb-hero h1 {font-size:1.55rem; margin:0 0 4px 0; color:#fff;} .tb-hero p {margin:0; opacity:.9; font-size:.95rem;}
.tb-badge {display:inline-block; padding:2px 10px; border-radius:999px; background:rgba(255,255,255,.18); font-size:.78rem; margin-right:6px;}
.tb-status {border-radius:12px; padding:14px 18px; font-weight:600; font-size:1.05rem; margin:6px 0 12px 0;}
.tb-ok {background:#dcfce7; color:#14532d; border:1px solid #86efac;} .tb-review {background:#fef3c7; color:#78350f; border:1px solid #fcd34d;}
.tb-failed {background:#fee2e2; color:#7f1d1d; border:1px solid #fca5a5;}
.tb-card {border:1px solid rgba(128,128,128,.25); border-radius:12px; padding:12px 16px; margin-bottom:10px;}
.tb-bar {height:12px; border-radius:6px; background:#0e7490;} .tb-barbg {background:rgba(128,128,128,.18); border-radius:6px; width:100%;}
.tb-small {font-size:.82rem; opacity:.8;}
</style>
"""

def _prob_bars(st, probs, top):
    html = ""
    for c, p in sorted(probs.items(), key=lambda kv: -kv[1]):
        b = "font-weight:700;" if c == top else ""
        html += (f"<div style='display:flex;align-items:center;gap:10px;margin:5px 0;{b}'><div style='width:120px;flex-shrink:0'>{c.replace('_', ' ')}</div>"
                 f"<div class='tb-barbg'><div class='tb-bar' style='width:{max(1, 100 * p):.1f}%'></div></div><div style='width:64px;flex-shrink:0;text-align:right'>{100 * p:.1f}%</div></div>")
    st.markdown(html, unsafe_allow_html=True)

def _result_block(st, name, data, r, pred, card, mask_alpha, show_mask, cam_alpha):
    status = r.get("status", "failed")
    if status == "failed":
        st.markdown(f"<div class='tb-status tb-failed'>Could not analyse this image. {'; '.join(r.get('reasons', []))}</div>", unsafe_allow_html=True); return
    if status == "ok":
        st.markdown(f"<div class='tb-status tb-ok'>Answer: {r['label'].replace('_', ' ')} &nbsp;·&nbsp; calibrated confidence {100 * r['confidence']:.1f}%"
                    f"<br><span class='tb-small'>All safety checks passed (confidence threshold, out-of-distribution check, classifier-segmenter agreement).</span></div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='tb-status tb-review'>Referred to a clinician - the model does not answer this image on its own."
                    f"<br><span class='tb-small'>Reason: {'; '.join(r.get('reasons', []))}</span></div>", unsafe_allow_html=True)
    P = panels(r, mask_alpha, show_mask, cam_alpha)
    c0, c1, c2, c3 = st.columns(4)
    c0.image(original_rgb(data), caption="Uploaded image", width="stretch")
    for col, (t, a) in zip((c1, c2, c3), P.items()): col.image(a, caption=t, width="stretch")
    a, b = st.columns([1.15, 1])
    with a:
        st.markdown("<div class='tb-card'><b>Class probabilities</b> (temperature-calibrated)</div>", unsafe_allow_html=True)
        if status == "review": st.caption("Shown for transparency only: this is the model's best guess, not an answer.")
        _prob_bars(st, r["probabilities"], r["label"])
    with b:
        cal = pred.meta["calibration"]
        st.markdown("<div class='tb-card'><b>Measurements</b></div>", unsafe_allow_html=True)
        if r.get("label") != "No_tumor":
            st.metric("Tumour area (224 x 224)", f"{r['tumour_area_px_224']} px"); st.caption(f"about {100 * head_fraction(r):.1f}% of the head region")
        else:
            st.metric("Tumour area", "none predicted")
        st.metric("Out-of-distribution distance", f"{r['ood_distance']:.2f}"); st.caption(f"images above {cal['ood_threshold']:.2f} are referred (validation 99th percentile)")
        thr = cal.get("review_threshold")
        st.caption(f"Answer threshold: {thr if thr is not None else 'none (every image is referred)'} - the lowest confidence at which validation proves "
                   f">= {100 * cal.get('selective_target_acc', 0.95):.0f}% accuracy on answered images.")
    d1, d2 = st.columns(2)
    d1.download_button("Download PDF report", report_pdf(name, data, r, pred, card), file_name=f"{Path(name).stem}_report.pdf", mime="application/pdf", width="stretch")
    js = {k: v for k, v in r.items() if k not in ("mask_224", "cam_14", "canonical_image_224")}
    d2.download_button("Download result (JSON)", json.dumps(js, indent=2, default=float), file_name=f"{Path(name).stem}_result.json", mime="application/json", width="stretch")

def main():
    import streamlit as st
    import pandas as pd
    st.set_page_config(page_title="TinyBrainNet - brain MRI analysis", page_icon="🧠", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    pred = st.cache_resource(load_predictor)(); card = load_card(); meta = pred.meta
    st.markdown(f"""<div class='tb-hero'><h1>TinyBrainNet · brain tumour MRI analysis</h1>
      <p>Classifies a 2D MRI slice (glioma / meningioma / pituitary / no tumour), outlines the tumour, and refers every uncertain or unusual image to a clinician.</p>
      <div style='margin-top:8px'><span class='tb-badge'>{meta['model'].get('params_trainable', 0):,} parameters</span><span class='tb-badge'>CPU</span>
      <span class='tb-badge'>research prototype - not a medical device</span></div></div>""", unsafe_allow_html=True)
    with st.sidebar:
        st.header("Display")
        show_mask = st.toggle("Show tumour segmentation", True); mask_alpha = st.slider("Segmentation opacity", 0.1, 0.9, 0.45, 0.05)
        cam_alpha = st.slider("Attention-map opacity", 0.1, 0.9, 0.45, 0.05)
        st.divider(); st.header("Model")
        for h in card.get("headline", [])[:6]: st.markdown(f"**{h['claim']}**  \n{h['value_text']}")
        st.caption(f"Weights SHA-256 {meta['model']['weights_sha256'][:16]}… · selected arm {meta['training']['selected_arm']} (validation only)")
        st.divider(); st.caption("Upload only de-identified images. Nothing is stored by this app.")
    t1, t2, t3, t4 = st.tabs(["Single image", "Batch", "Model card & evaluation", "About & limitations"])
    with t1:
        up = st.file_uploader("Upload one MRI slice", type=list(EXT), key="single")
        if up is None:
            st.info("Upload a de-identified 2D brain MRI slice (JPG / PNG / BMP / TIFF). The app answers only when its safety checks pass.")
        else:
            data = up.getvalue()
            with st.spinner("Analysing..."): r = analyse(pred, data, up.name)
            _result_block(st, up.name, data, r, pred, card, mask_alpha, show_mask, cam_alpha)
    with t2:
        ups = st.file_uploader("Upload several slices", type=list(EXT), accept_multiple_files=True, key="batch")
        if ups:
            rows, pdfs, bar = [], {}, st.progress(0.0, text="Analysing...")
            for i, u in enumerate(ups):
                d = u.getvalue(); r = analyse(pred, d, u.name); rows.append(summary_row(u.name, r)); pdfs[u.name] = report_pdf(u.name, d, r, pred, card)
                bar.progress((i + 1) / len(ups), text=f"{i + 1} / {len(ups)}")
            df = pd.DataFrame(rows); n_ok = int((df.status == "ok").sum())
            c1, c2, c3 = st.columns(3); c1.metric("Images", len(df)); c2.metric("Answered", n_ok); c3.metric("Referred / failed", len(df) - n_ok)
            st.dataframe(df, width="stretch", hide_index=True)
            zb = io.BytesIO()
            with zipfile.ZipFile(zb, "w", zipfile.ZIP_DEFLATED) as z:
                for n, b in pdfs.items(): z.writestr(f"{Path(n).stem}_report.pdf", b)
                z.writestr("summary.csv", df.to_csv(index=False))
            e1, e2 = st.columns(2)
            e1.download_button("Download summary (CSV)", df.to_csv(index=False), "tinybrainnet_batch_summary.csv", "text/csv", width="stretch")
            e2.download_button("Download all reports (ZIP)", zb.getvalue(), "tinybrainnet_batch_reports.zip", "application/zip", width="stretch")
    with t3:
        if not card: st.warning("model_card.json is missing.")
        else:
            st.subheader("Headline results (read from the evaluation files of the training notebook)")
            for h in card.get("headline", []): st.markdown(f"- **{h['claim']}**: {h['value_text']}  <span class='tb-small'>({h.get('source', '')})</span>", unsafe_allow_html=True)
            if card.get("selective_by_tier"):
                st.subheader("How the app answers on each test set")
                st.dataframe(pd.DataFrame(card["selective_by_tier"]), width="stretch", hide_index=True)
                st.caption("Answered = share of images the app answers on its own; accuracy is on those images. All thresholds were fixed on validation data before any test set was used.")
            if card.get("comparisons"):
                st.subheader("Compared with the earlier models (same images, exact McNemar test)")
                st.dataframe(pd.DataFrame(card["comparisons"]), width="stretch", hide_index=True)
            for f in card.get("figures", []):
                p = ROOT / f
                if p.exists(): st.image(str(p), caption=Path(f).stem.replace("_", " "), width="stretch")
            if card.get("web_test"): st.subheader("Web / user test images"); st.json(card["web_test"])
            st.subheader("Training and verification"); st.json({k: card[k] for k in ("training", "verification") if k in card})
    with t4:
        st.markdown(card.get("limitations_md", "") or "Research prototype.")

if __name__ == "__main__":
    main()
