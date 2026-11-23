import base64
import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import streamlit as st
from PIL import Image
from matplotlib import colormaps

from src.inference import load_model, predict_with_cam, CLASSES

REPO_URL = "https://github.com/YOUR_GITHUB_USERNAME/Medical-Vision-AI-XAI"

st.set_page_config(page_title="Explainable Brain MRI Classifier", page_icon="🧠", layout="wide")

NICE = {"glioma": "Glioma", "meningioma": "Meningioma",
        "notumor": "No tumor", "pituitary": "Pituitary tumor"}
SHORT = {"glioma": "Glioma", "meningioma": "Meningioma",
         "notumor": "No tumor", "pituitary": "Pituitary"}
MODELS = {"ResNet-50 (23.5M parameters)": "resnet50",
          "EfficientNet-B0 (4.0M parameters)": "efficientnet_b0"}
SAMPLE_DIR = ROOT / "app" / "samples"


def _order(p):
    parts = p.stem.split("_")
    cls = CLASSES.index(parts[0]) if parts[0] in CLASSES else 9
    return ("MISSED" in parts, cls, p.stem)


SAMPLES = sorted(SAMPLE_DIR.glob("*.jpg"), key=_order)

# ---------------------------------------------------------------- styling
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,400;6..72,600&family=Instrument+Sans:wght@400;500;600&display=swap');
:root{
  --bg:#F2F5F7; --paper:#FFFFFF; --ink:#14212B; --muted:#5B6B78; --line:#DCE3E8;
  --teal:#0F7C80; --teal-soft:#E4F1F1; --amber:#B7791F; --amber-soft:#FFF5E0;
  --red:#B3261E; --red-soft:#FDECEA; --slate:#C5D0D8;
}
.stApp{background:var(--bg);}
.stApp, .stApp p, .stApp label, .stApp button, .stApp input, .stApp li{
  font-family:'Instrument Sans',system-ui,-apple-system,'Segoe UI',sans-serif;
}
[data-testid="stToolbar"], [data-testid="stDecoration"], footer{display:none !important;}
.block-container{max-width:1180px; padding-top:2.4rem; padding-bottom:4rem;}
[data-testid="stSidebar"]{background:var(--paper); border-right:1px solid var(--line);}

.hero{margin:0 0 1.4rem;}
.hero h1.title{
  font-family:'Newsreader',Georgia,serif; font-weight:600; color:var(--ink);
  font-size:clamp(2.1rem,4.4vw,3.2rem); line-height:1.06; letter-spacing:-0.012em;
  margin:0 0 .55rem; padding:0;
}
.hero p.lede{font-size:1.08rem; line-height:1.55; color:var(--muted); max-width:62ch; margin:0 0 1.25rem;}
.facts{display:flex; flex-wrap:wrap; gap:1.1rem 2.2rem;}
.fact{border-left:2px solid var(--teal); padding-left:.85rem; max-width:230px;}
.fact b{display:block; font-family:'Newsreader',Georgia,serif; font-size:1.55rem; font-weight:600; color:var(--ink); line-height:1.1;}
.fact span{display:block; font-size:.86rem; color:var(--muted); line-height:1.35; margin-top:.2rem;}

.banner{background:var(--amber-soft); border-left:4px solid var(--amber); border-radius:10px;
  padding:.7rem 1rem; margin:1.3rem 0 1.6rem; color:#5C3D0B; font-size:.93rem; line-height:1.45;}

h2.sec{font-family:'Newsreader',Georgia,serif; font-weight:600; font-size:1.55rem; color:var(--ink);
  margin:1.6rem 0 .5rem; padding:0; letter-spacing:-0.005em;}

.thumb{border-radius:10px; overflow:hidden; border:2px solid transparent; background:#0E1A24; line-height:0;}
.thumb.sel{border-color:var(--teal);}
.thumb img{width:100%; aspect-ratio:1; object-fit:cover; display:block;}
.stButton>button{width:100%; border-radius:10px; font-weight:500; margin-top:.35rem;}

.stage{background:linear-gradient(180deg,#112130,#0A1520); border-radius:18px; padding:18px 18px 14px;
  box-shadow:0 18px 40px -24px rgba(10,25,40,.6);}
.panes{display:grid; grid-template-columns:1fr 1fr; gap:14px;}
.panes figure{margin:0;}
.panes img{width:100%; aspect-ratio:1; object-fit:cover; border-radius:10px; display:block; background:#000;}
.panes figcaption{color:#B6C5D1; font-size:.86rem; margin-top:.5rem; line-height:1.35;}
.legend{display:flex; align-items:center; gap:10px; margin-top:.9rem; color:#8EA2B2; font-size:.78rem;}
.ramp{flex:1; height:8px; border-radius:4px;
  background:linear-gradient(90deg,#00007f,#0000ff,#00ffff,#ffff00,#ff0000,#7f0000);}
.stage-note{color:#8EA2B2; font-size:.82rem; line-height:1.45; margin:.7rem 0 0;}
.chips{margin-top:.8rem; display:flex; gap:.5rem; flex-wrap:wrap;}
.chip{background:var(--teal-soft); color:#0B5A5C; border-radius:999px; padding:.22rem .75rem; font-size:.84rem; font-weight:500;}
.chip.fail{background:var(--amber-soft); color:#7A4F0C;}

.card{background:var(--paper); border:1px solid var(--line); border-radius:18px; padding:1.4rem 1.5rem;}
.res-name{font-family:'Newsreader',Georgia,serif; font-weight:600; font-size:2.5rem; line-height:1.05;
  color:var(--ink); margin:.1rem 0 .7rem; letter-spacing:-0.01em;}
.conf-row{display:flex; align-items:baseline; gap:.5rem; margin-bottom:.5rem;}
.conf-num{font-size:1.6rem; font-weight:600; color:var(--ink);}
.conf-txt{font-size:.92rem; color:var(--muted);}
.meter{height:10px; background:#E6ECF0; border-radius:5px; overflow:hidden;}
.meter i{display:block; height:100%; border-radius:5px;}
.tier{font-size:.9rem; line-height:1.5; margin:.8rem 0 0; color:var(--muted);}
.bars{margin-top:1.2rem; display:grid; gap:.55rem;}
.brow{display:grid; grid-template-columns:104px 1fr 54px; align-items:center; gap:.7rem; font-size:.9rem; color:var(--ink);}
.brow .track{height:8px; background:#E6ECF0; border-radius:4px; overflow:hidden;}
.brow .track i{display:block; height:100%; border-radius:4px; background:var(--slate);}
.brow.top .track i{background:var(--teal);}
.brow .pct{text-align:right; font-variant-numeric:tabular-nums; color:var(--muted);}
.brow.top .pct{color:var(--ink); font-weight:600;}
.note{background:var(--amber-soft); border-left:4px solid var(--amber); border-radius:10px;
  padding:.75rem 1rem; margin-top:1.1rem; color:#5C3D0B; font-size:.88rem; line-height:1.5;}
.alert{background:var(--red-soft); border-left:4px solid var(--red); border-radius:10px;
  padding:.8rem 1rem; margin:0 0 1rem; color:#6B1511; font-size:.95rem; line-height:1.5;}
.side-about{font-size:.86rem; line-height:1.55; color:var(--muted);}
@media (max-width:640px){ .panes{grid-template-columns:1fr;} }
</style>
"""


def html(s):
    """Strip indentation so Markdown never turns the HTML into a code block."""
    return "\n".join(line.strip() for line in s.strip().splitlines())


# ---------------------------------------------------------------- helpers
@st.cache_resource(show_spinner="Loading model...")
def get_model(name):
    return load_model(name, models_dir=str(ROOT / "models"))


def b64_jpeg(img, size, quality=88):
    buf = io.BytesIO()
    img.convert("RGB").resize((size, size)).save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


@st.cache_data(show_spinner=False)
def thumb_b64(path_str):
    return b64_jpeg(Image.open(path_str), 200, 80)


def overlay(image, cam, alpha):
    size = 448
    base = np.asarray(image.convert("RGB").resize((size, size))).astype(np.float32) / 255
    cam_img = Image.fromarray((cam * 255).astype(np.uint8)).resize((size, size), Image.BILINEAR)
    heat = colormaps["jet"](np.asarray(cam_img).astype(np.float32) / 255)[..., :3]
    mixed = (1 - alpha) * base + alpha * heat
    return Image.fromarray((mixed * 255).astype(np.uint8))


def mri_warning(image):
    """Cheap sanity check. A heuristic, not a real out-of-distribution detector."""
    rgb = np.asarray(image.convert("RGB").resize((224, 224))).astype(np.float32)
    colorfulness = (rgb.max(axis=2) - rgb.min(axis=2)).mean()
    dark_share = (rgb.mean(axis=2) < 25).mean()
    reasons = []
    if colorfulness > 20:
        reasons.append("it has noticeable color (MRI scans are gray)")
    if dark_share < 0.08:
        reasons.append("it has almost no dark background")
    return reasons


def pretty(stem):
    parts = stem.split("_")
    if "MISSED" in parts:
        return f"Failure case {parts[-1]}", True
    return f"{SHORT[parts[0]]} {parts[-1]}", False


def pick_sample(stem):
    st.session_state.sample = stem
    st.session_state.src = "sample"


def use_upload():
    st.session_state.src = "upload"


st.session_state.setdefault("src", "sample")
st.session_state.setdefault("sample", SAMPLES[0].stem if SAMPLES else None)

st.markdown(CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("### Settings")
    model_label = st.selectbox("Model", list(MODELS))
    model_name = MODELS[model_label]
    explain_choice = st.selectbox("Explain which class?",
                                  ["Predicted class"] + [NICE[c] for c in CLASSES])
    alpha = st.slider("Heatmap strength", 10, 90, 45, step=5) / 100
    st.markdown(html(f"""
    <p class="side-about">Two ImageNet-pretrained CNNs fine-tuned on a deduplicated brain MRI
    dataset, with Grad-CAM written from scratch.
    <a href="{REPO_URL}" target="_blank">Code and write-up on GitHub</a>.</p>
    """), unsafe_allow_html=True)

# ---------------------------------------------------------------- hero
st.markdown(html("""
<div class="hero">
  <h1 class="title">Explainable Brain MRI Classifier</h1>
  <p class="lede">Classifies a scan as glioma, meningioma, pituitary tumor or no tumor, and shows
  where the model looked. Built to ask a harder question than accuracy: when can we trust it?</p>
  <div class="facts">
    <div class="fact"><b>95.1%</b><span>accuracy on a leak-free test set of 1,180 scans (ResNet-50)</span></div>
    <div class="fact"><b>23%</b><span>of the original test images had a copy in the training set</span></div>
    <div class="fact"><b>37 of 377</b><span>gliomas were labeled &ldquo;no tumor&rdquo; by both models</span></div>
  </div>
</div>
<div class="banner"><b>Research demo only.</b> Not a medical device and not medical advice.
Do not use it to make decisions about health or treatment.</div>
"""), unsafe_allow_html=True)

# ---------------------------------------------------------------- choose an image
st.markdown('<h2 class="sec">Choose a scan</h2>', unsafe_allow_html=True)
tab_sample, tab_upload = st.tabs(["Sample scans", "Upload your own"])

with tab_sample:
    if not SAMPLES:
        st.info("No sample images found in app/samples.")
    per_row = 5
    for start in range(0, len(SAMPLES), per_row):
        cols = st.columns(per_row)
        for col, p in zip(cols, SAMPLES[start:start + per_row]):
            label, failure = pretty(p.stem)
            selected = st.session_state.src == "sample" and st.session_state.sample == p.stem
            with col:
                cls = "thumb sel" if selected else "thumb"
                st.markdown(f'<div class="{cls}"><img src="data:image/jpeg;base64,{thumb_b64(str(p))}" alt="{label}"></div>',
                            unsafe_allow_html=True)
                st.button(label, key=f"pick_{p.stem}", on_click=pick_sample, args=(p.stem,),
                          type="primary" if selected else "secondary")

with tab_upload:
    st.file_uploader("Upload a brain MRI (JPG or PNG)", type=["jpg", "jpeg", "png"],
                     key="uploader", on_change=use_upload)
    st.caption("Your image is processed in memory and not stored.")

# ---------------------------------------------------------------- resolve the image
image, sample_stem = None, None
up = st.session_state.get("uploader")
if st.session_state.src == "upload" and up is not None:
    try:
        image = Image.open(io.BytesIO(up.getvalue())).convert("RGB")
    except Exception:
        st.error("This file could not be read as an image. Try a JPG or PNG.")
elif SAMPLES:
    sample_stem = st.session_state.sample
    image = Image.open(SAMPLE_DIR / f"{sample_stem}.jpg").convert("RGB")

if image is None:
    st.info("Pick a sample scan or upload an MRI to see a prediction.")
    st.stop()

# ---------------------------------------------------------------- input check
reasons = mri_warning(image)
if reasons:
    st.markdown(html("""
    <div class="alert"><b>This does not look like a brain MRI:</b> """ + "; ".join(reasons) + """.
    The model can only choose between its four classes and cannot say &ldquo;this is not a scan&rdquo;,
    so any answer for this image would be meaningless.</div>
    """), unsafe_allow_html=True)
    st.image(image.resize((280, 280)), caption="Your image")
    if not st.checkbox("Show the model's output anyway (it will not be meaningful)"):
        st.stop()

# ---------------------------------------------------------------- predict
model = get_model(model_name)
k_req = None if explain_choice == "Predicted class" else [NICE[c] for c in CLASSES].index(explain_choice)

with st.spinner("Analyzing the scan..."):
    cam, k, probs = predict_with_cam(model, model_name, image, class_idx=k_req)
pred = int(np.argmax(probs))
conf = float(probs[pred])

if conf >= 0.8:
    tier_color = "#0F7C80"
    tier_txt = ("Confident. Confidence is not a guarantee: in our tests, 37 to 47% of wrong answers "
                "came with more than 80% confidence.")
elif conf >= 0.6:
    tier_color = "#B7791F"
    tier_txt = "Moderate confidence. Treat this result with caution."
else:
    tier_color = "#B3261E"
    tier_txt = "Low confidence. The model is unsure, so this result is unreliable."

# ---------------------------------------------------------------- results
st.markdown('<h2 class="sec">What the model sees</h2>', unsafe_allow_html=True)
left, right = st.columns([1.6, 1], gap="large")

with left:
    scan_b64 = b64_jpeg(image, 448)
    heat_b64 = b64_jpeg(overlay(image, cam, alpha), 448)
    chips = ""
    if sample_stem and st.session_state.src == "sample":
        truth = sample_stem.split("_")[0]
        chips += f'<span class="chip">Dataset label: {NICE[truth]}</span>'
        if "MISSED" in sample_stem:
            chips += '<span class="chip fail">Known failure case of ResNet-50</span>'
    explained = NICE[CLASSES[k]]
    shown_note = "" if k == pred else f" This is not the model's answer ({NICE[CLASSES[pred]]})."
    st.markdown(html(f"""
    <div class="stage">
      <div class="panes">
        <figure><img src="data:image/jpeg;base64,{scan_b64}" alt="Input scan"><figcaption>Input scan</figcaption></figure>
        <figure><img src="data:image/jpeg;base64,{heat_b64}" alt="Grad-CAM heatmap"><figcaption>Evidence for {explained}</figcaption></figure>
      </div>
      <div class="legend"><span>less influence</span><div class="ramp"></div><span>more influence</span></div>
      <p class="stage-note">Warm areas pushed the model toward {explained}.{shown_note} The map is coarse
      (7&times;7 grid) and shows where evidence came from, not how the model reasoned.</p>
    </div>
    <div class="chips">{chips}</div>
    """), unsafe_allow_html=True)

with right:
    order = np.argsort(-probs)
    rows = ""
    for i in order:
        top = "top" if i == pred else ""
        rows += (f'<div class="brow {top}"><span>{NICE[CLASSES[i]]}</span>'
                 f'<div class="track"><i style="width:{probs[i]*100:.1f}%"></i></div>'
                 f'<span class="pct">{probs[i]:.1%}</span></div>')
    caution = ""
    if CLASSES[pred] == "notumor":
        caution = ('<div class="note"><b>&ldquo;No tumor&rdquo; is not reassurance here.</b> '
                   'In testing, 37 of 377 gliomas (about 10%) were labeled &ldquo;no tumor&rdquo; '
                   'by each model, often with high confidence.</div>')
    st.markdown(html(f"""
    <div class="card">
      <div class="res-name">{NICE[CLASSES[pred]]}</div>
      <div class="conf-row"><span class="conf-num">{conf:.0%}</span><span class="conf-txt">confidence</span></div>
      <div class="meter"><i style="width:{conf*100:.1f}%; background:{tier_color}"></i></div>
      <p class="tier">{tier_txt}</p>
      <div class="bars">{rows}</div>
      {caution}
    </div>
    """), unsafe_allow_html=True)

# ---------------------------------------------------------------- details
st.markdown('<h2 class="sec">How to read this</h2>', unsafe_allow_html=True)
with st.expander("What the heatmap does and does not tell you"):
    st.markdown(
        "- **Grad-CAM shows where the evidence came from, not why.** A heatmap on the lesion is "
        "reassuring but does not prove the model is right. A heatmap on the skull or background is a warning sign.\n"
        "- **The explanation can look sensible for a meaningless input.** A photo of a handwritten page "
        "was classified \"no tumor\" with 100% confidence, and still received a heatmap.\n"
        "- Choose a different class in the sidebar to see where the evidence for *that* class lies."
    )

with st.expander("What this project found"):
    st.markdown(
        "- **The public dataset leaked.** 23% of the official test images had an exact copy in the training set; "
        "these were removed from training and all results use a leak-free test set (1,180 images).\n"
        "- **Accuracy is about 95%** for ResNet-50 (95.1%) and EfficientNet-B0 (94.5%), a difference within noise. "
        "EfficientNet-B0 is about 6 times smaller.\n"
        "- **Glioma recall is only 84 to 86%.** Validation had hidden this (99.6%). The misses are all glioma scans "
        "stored as color files, a hint that image source matters, which is plausible but not proven.\n"
        "- **Both models are fragile.** Mild blur cut accuracy to 54 to 65%.\n"
        "- **Limits:** no patient IDs, no external validation, single training run, no radiologist review.\n"
        "- Data: Brain Tumor MRI Dataset by Masoud Nickparvar (CC BY 4.0)."
    )