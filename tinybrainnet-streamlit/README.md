# TinyBrainNet - brain tumour MRI analysis (research prototype)

A 496,503-parameter network that classifies a 2D brain MRI slice
(glioma / meningioma / pituitary / no tumour), outlines the tumour, and refers uncertain or unusual images to a clinician.
**Not a medical device. Not for diagnosis.**

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Create an empty GitHub repository and upload **the contents of this folder** (app.py at the top level; keep `bundle/`, `assets/`, `.streamlit/`).
2. Go to share.streamlit.io -> **Create app** -> choose the repository, branch `main`, main file `app.py`.
3. Under *Advanced settings* choose Python 3.11 or 3.12, then **Deploy**. The first start installs CPU PyTorch and takes a few minutes.

## What is inside
| Path | Content |
|---|---|
| `app.py` | Streamlit interface (single image, batch, model card, limitations) |
| `report.py` | one-page PDF report per image |
| `bundle/` | the verified deployment bundle: `model.py`, `preprocess.py` (identical to training), `infer.py`, weights, calibration, OOD statistics |
| `model_card.json` | every reported number, read from the training notebook's saved evaluation files |
| `assets/` | evaluation figures |

Weights SHA-256: `7af222c95c9f1ede69cf78d21f29f8edf14ff63c242370e1fa9c606cf93b17cb`
