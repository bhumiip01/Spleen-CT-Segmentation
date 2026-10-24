import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import streamlit as st

GT_CMAP = ListedColormap(["#2ca02c"])  # green  (ground truth)
PR_CMAP = ListedColormap(["#d62728"])  # red    (prediction)

st.set_page_config(page_title="Spleen CT Segmentation", layout="wide")


@st.cache_resource
def load_model():
    m = torch.jit.load("model_scripted.pt", map_location="cpu")
    m.eval()
    return m


@st.cache_data
def load_samples():
    d = np.load("samples.npz", allow_pickle=True)
    cases = [str(c) for c in d["__cases"]]
    data = {}
    for name in cases:
        data[name] = {
            "ct": d[f"{name}__ct"].astype(np.float32),
            "gt": d[f"{name}__gt"],
            "dice": float(d[f"{name}__dice"]),
            "slices": d[f"{name}__slices"],
        }
    return cases, data


model = load_model()
cases, data = load_samples()

st.title("Spleen Segmentation from CT — 2D U-Net")
st.markdown(
    "A 2D U-Net (MONAI) segmenting the spleen from abdominal CT — Medical Segmentation Decathlon, Task09. "
    "Pick a held-out **test** patient to see the model's prediction (red) against the ground truth (green).\n\n"
    "Mean test Dice is **0.74** across 6 held-out patients. This is a proof-of-concept: the model "
    "over-predicts on small spleens — a known failure, analyzed in the "
    "[GitHub README](https://github.com/bhumiip01/Spleen-CT-Segmentation)."
)

default = cases.index("spleen_20") if "spleen_20" in cases else 0
patient = st.selectbox("Held-out test patient", cases, index=default)

rec = data[patient]
ct, gt, slices = rec["ct"], rec["gt"], rec["slices"]

preds = []
with torch.no_grad():
    for i in range(len(ct)):
        xi = torch.from_numpy(ct[i])[None, None]          # (1, 1, 256, 256)
        pi = (torch.sigmoid(model(xi)) > 0.5).float().numpy()[0, 0]
        preds.append(pi)
pred = np.stack(preds)

st.markdown(
    f"### {patient}\n"
    f"Full-volume test Dice: **{rec['dice']:.3f}**  —  prediction in **red**, ground truth in **green**."
)

fig, ax = plt.subplots(len(ct), 3, figsize=(9, 3 * len(ct)))
ax = np.atleast_2d(ax)
for r in range(len(ct)):
    base = ct[r].T
    ax[r, 0].imshow(base, cmap="gray", origin="lower")
    ax[r, 0].set_title(f"CT (slice {int(slices[r])})")
    ax[r, 1].imshow(base, cmap="gray", origin="lower")
    ax[r, 1].imshow(np.ma.masked_where(gt[r].T == 0, gt[r].T), cmap=GT_CMAP, alpha=0.55, origin="lower")
    ax[r, 1].set_title("ground truth")
    ax[r, 2].imshow(base, cmap="gray", origin="lower")
    ax[r, 2].imshow(np.ma.masked_where(pred[r].T == 0, pred[r].T), cmap=PR_CMAP, alpha=0.55, origin="lower")
    ax[r, 2].set_title("prediction")
    for c in range(3):
        ax[r, c].axis("off")
plt.tight_layout()
st.pyplot(fig)

st.caption(
    "Model runs live in the browser session on bundled sample slices. "
    "Dice shown is the full held-out-volume score computed during evaluation."
)
