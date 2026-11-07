"""PDF report for one analysed image (fpdf2, core fonts -> ASCII text only)."""
import io, datetime, hashlib, uuid
import numpy as np
from PIL import Image
from fpdf import FPDF

def _ascii(s):
    return str(s).encode("ascii", "replace").decode("ascii")

def _img(arr):
    b = io.BytesIO(); Image.fromarray(arr).save(b, format="PNG"); b.seek(0); return b

class _PDF(FPDF):
    def header(self):
        self.set_fill_color(14, 116, 144); self.rect(0, 0, 210, 18, "F")
        self.set_text_color(255, 255, 255); self.set_font("Helvetica", "B", 13); self.set_xy(10, 5)
        self.cell(0, 8, "TinyBrainNet - brain MRI analysis report (research prototype)")
        self.set_text_color(0, 0, 0); self.ln(16)
    def footer(self):
        self.set_y(-14); self.set_font("Helvetica", "I", 7); self.set_text_color(110, 110, 110)
        self.cell(0, 5, _ascii("Not a medical device. Not for diagnosis. Confidence is not clinical certainty. Page %d" % self.page_no()), align="C")

def make_pdf(file_name, image_bytes, result, panels, meta, card=None):
    """panels: dict title -> RGB uint8 array (shown side by side). Returns the PDF as bytes."""
    pdf = _PDF(format="A4"); pdf.set_auto_page_break(True, 16); pdf.add_page()
    pdf.set_font("Helvetica", "", 9)
    rid = uuid.uuid4().hex[:12].upper(); now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sha = hashlib.sha256(image_bytes).hexdigest()
    for k, v in (("Report ID", rid), ("Created", now), ("File", file_name), ("Image SHA-256", sha[:32] + "...")):
        pdf.set_font("Helvetica", "B", 9); pdf.cell(32, 5, _ascii(k)); pdf.set_font("Helvetica", "", 9); pdf.cell(0, 5, _ascii(v), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    status = result.get("status", "failed")
    colour = {"ok": (22, 163, 74), "review": (217, 119, 6)}.get(status, (220, 38, 38))
    text = {"ok": "ANSWERED - all safety checks passed",
            "review": "REFERRED - the model does not answer this image on its own; a clinician must review it"}.get(status, "FAILED - the image could not be analysed")
    pdf.set_fill_color(*colour); pdf.set_text_color(255, 255, 255); pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 9, _ascii("  " + text), fill=True, new_x="LMARGIN", new_y="NEXT"); pdf.set_text_color(0, 0, 0); pdf.ln(3)
    if panels:
        w = (190 - 4 * (len(panels) - 1)) / len(panels); x0, y0 = pdf.get_x(), pdf.get_y()
        for i, (title, arr) in enumerate(panels.items()):
            x = x0 + i * (w + 4); pdf.image(_img(arr), x=x, y=y0, w=w, h=w)
            pdf.set_xy(x, y0 + w + 1); pdf.set_font("Helvetica", "", 8); pdf.multi_cell(w, 4, _ascii(title), align="C")
        pdf.set_xy(x0, y0 + w + 10)
    if status != "failed":
        pdf.set_font("Helvetica", "B", 11); pdf.cell(0, 7, "Result", new_x="LMARGIN", new_y="NEXT"); pdf.set_font("Helvetica", "", 9)
        lab = result.get("label", "-"); conf = result.get("confidence", 0.0)
        pdf.cell(0, 5, _ascii(("Prediction: %s (calibrated confidence %.1f %%)" if status == "ok" else
                               "Model's best guess (NOT an answer): %s (calibrated confidence %.1f %%)") % (lab, 100 * conf)), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(1)
        for c, p in sorted(result.get("probabilities", {}).items(), key=lambda kv: -kv[1]):
            pdf.cell(32, 5, _ascii(c)); pdf.set_fill_color(14, 116, 144); pdf.cell(max(0.5, 100 * p), 4, "", fill=True)
            pdf.set_x(pdf.l_margin + 32 + 102); pdf.cell(0, 5, "%.1f %%" % (100 * p), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(1)
        pdf.cell(0, 5, _ascii("Segmented tumour area: %d px at 224 x 224 (shown only when the classifier predicts a tumour)" % result.get("tumour_area_px_224", 0)), new_x="LMARGIN", new_y="NEXT")
        cal = meta.get("calibration", {})
        pdf.cell(0, 5, _ascii("Out-of-distribution distance: %.2f (limit %.2f)" % (result.get("ood_distance", 0.0), cal.get("ood_threshold", float("nan")))), new_x="LMARGIN", new_y="NEXT")
    if result.get("reasons"):
        pdf.ln(2); pdf.set_font("Helvetica", "B", 10); pdf.cell(0, 6, "Why the image was referred", new_x="LMARGIN", new_y="NEXT"); pdf.set_font("Helvetica", "", 9)
        for r in result["reasons"]: pdf.multi_cell(0, 5, _ascii("- " + r), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2); pdf.set_font("Helvetica", "B", 10); pdf.cell(0, 6, "Model", new_x="LMARGIN", new_y="NEXT"); pdf.set_font("Helvetica", "", 8)
    m = meta.get("model", {}); t = meta.get("training", {}); cal = meta.get("calibration", {})
    thr = cal.get("review_threshold"); pct = int(100 * cal.get("selective_target_acc", 0.95))
    thr_txt = ("answer threshold %.3f (proves >= %d %% accuracy on validation)" % (thr, pct)) if thr is not None else \
              ("no threshold proved >= %d %% accuracy on validation, so every image is referred" % pct)
    lines = ["TinyBrainNetP21, %s trainable parameters; weights SHA-256 %s..." % (format(m.get("params_trainable", 0), ","), str(m.get("weights_sha256", ""))[:16]),
             "Selected arm %s (plan %s) by validation only; temperature %.3f; %s" % (t.get("selected_arm"), t.get("plan"), cal.get("temperature", 1.0), thr_txt)]
    if card and card.get("headline"):
        for h in card["headline"][:4]: lines.append("%s: %s" % (h["claim"], h["value_text"]))
    for l in lines: pdf.multi_cell(0, 4.5, _ascii(l), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2); pdf.set_fill_color(254, 243, 199); pdf.set_font("Helvetica", "", 8)
    pdf.multi_cell(0, 4.5, _ascii("Intended use: " + meta.get("intended_use", "research prototype") + ". The model sees one 2D slice, not the whole study. "
                                  "A referral is not an error: the model refers every image for which validation could not prove >= 95 % accuracy, "
                                  "images unlike its training data, colour images, and cases where classification and segmentation disagree."), fill=True)
    return bytes(pdf.output())
