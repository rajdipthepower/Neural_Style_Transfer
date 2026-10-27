# DeepArt: Neural Style Transfer in PyTorch

Two implementations of neural style transfer, plus a **live web app**:

1. **Gatys et al. (optimization-based):** high quality, slow, with documented hyperparameter tuning.
2. **AdaIN (feed-forward):** arbitrary style transfer in a single pass, with runtime controls.
3. **DeepArt web app:** the AdaIN pipeline deployed as a Streamlit app.

🔗 **Live demo:** `https://neuralstyletransfer-by-raj.streamlit.app/` *(the app sleeps after a period of inactivity; the first load may take a moment)*

---

## 📁 Repository Structure

| Folder | Contents |
|---|---|
| `Gaty's Optimization/` | Gatys optimization notebook, results, loss curve |
| `adain_style_transfer/` | AdaIN experiment notebook and result images |
| `Data/` | Sample content and style images |
| `Deployment/` | The web app (Streamlit, plus the original Flask version) |

---

## 🌐 DeepArt Web App

Upload a photo, pick a painting style, and get a stylized result in seconds on CPU.

**Features**
- 14 built-in public-domain styles grouped as **Portrait**, **Landscape** and **Abstract**, or upload your own style image.
- **Style strength** slider (α) and **brushstroke size** (Fine / Medium / Broad).
- **Mix two styles**, each with its own brushstroke size and a blend slider.
- Compare with the original, and download the result.
- Input validation: file type, size (10 MB), minimum and maximum dimensions, corrupt-image detection.

**Design decisions**
- Style statistics (channel mean and std) are **precomputed** for every built-in style at 128 / 512 / 1280 px, so a request never has to encode a large style image.
- Models and statistics load once and are cached. One stylization runs at a time, to stay within memory limits.
- Content is capped at 640 px on the long edge, and custom styles at 512 px, to keep memory around 600–750 MB.
- A Flask API with a plain HTML/JS frontend was built first and is kept in the repo (`Deployment/backend/app.py`, `Deployment/frontend/`).

**Measured on a desktop CPU:** about 1.3–2 s per image warm, and 600–750 MB peak RAM including Streamlit.

### Run locally

```bash
cd Deployment
python -m venv .venv
.venv\Scripts\activate          # Windows (use source .venv/bin/activate on Linux/macOS)
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Optional, for the Flask version: `pip install flask`, then `python backend/app.py`.

### Add or change styles

1. Put images in `Deployment/styles/<portrait|landscape|abstract>/` (lowercase names with underscores).
2. From `Deployment/`, run `python build_styles.py` to regenerate thumbnails and `styles.json`.
3. Run `python backend/precompute_styles.py` to regenerate `styles/style_stats.pt`.

### Project layout (`Deployment/`)

```
backend/      inference.py (AdaIN pipeline), app.py (Flask API), precompute_styles.py
frontend/     Flask HTML/CSS/JS UI and background image
models/       vgg_normalised.pth, decoder.pth
styles/       style images, thumbnails, styles.json, style_stats.pt
streamlit_app.py   the deployed app
build_styles.py    thumbnails and styles.json generator
```

---

## 🔬 Part 1: Gatys Optimization Experiments

Neural style transfer must balance content preservation against style replication. Tuning process:

1. **Initial attempt (failed initialization)**
   - 500 epochs, `content_weight = 1`, `style_weight = 5e-6`, initialized with the **target (content) image**.
   - Optimization got trapped near the content image and produced minimal stylistic change.

2. **Second attempt (noise)**
   - 700 epochs, same weights, initialized with **random noise**.
   - Output was noisy and structurally incoherent: content and style were poorly balanced.

3. **Final render (optimized balance)**
   - 700 epochs, `content_weight = 200`, `style_weight = 1e-8`.
   - A higher content weight and lower style weight preserved the shapes of the photograph while transferring color and texture.

### Loss observations
- **Content loss:** starts very low when initialized with the content image, and can spike early under random initialization as structure is rebuilt from noise.
- **Style loss:** drops quickly within the first 100–200 epochs.
- **Saturation:** both losses flatten asymptotically, so further epochs give diminishing returns.

---

## ⚡ Part 2: Real-Time Arbitrary Style Transfer (AdaIN)

Gatys' method optimizes image pixels iteratively, which is expensive. **Adaptive Instance Normalization (AdaIN)** replaces that with a single feed-forward pass.

### Encoder–decoder pipeline
A frozen VGG-19 encoder (up to `relu4_1`) and a symmetric, reflection-padded decoder that reconstructs the image from feature space. Pretrained weights come from [naoto0804/pytorch-AdaIN](https://github.com/naoto0804/pytorch-AdaIN). The encoder's first 1×1 convolution handles the RGB→BGR swap and mean subtraction internally, so inputs are plain 0–1 RGB.

### Feature-space alignment
AdaIN aligns the channel-wise mean ($\mu$) and standard deviation ($\sigma$) of content features to those of the style features:

$$
\text{AdaIN}(f(c), f(s)) = \sigma(f(s)) \left( \frac{f(c)-\mu(f(c))}{\sigma(f(c))} \right) + \mu(f(s))
$$

Since only $\mu$ and $\sigma$ of the style are needed, they can be **precomputed** per style and scale.

### Runtime controls (no retraining)

**1. Style strength ($\alpha$)**

$$
F_{\text{output}} = (1-\alpha)\,f(c) + \alpha\,F_{\text{stylized}}, \quad \alpha \in [0,1]
$$

$\alpha = 0$ keeps the original content, and $\alpha = 1$ applies the full style.

**2. Multi-style blending**

$$
F_{\text{combined}} = \sum_{k=1}^{K} w_k \cdot \text{AdaIN}\bigl(f(c), f(s_k)\bigr), \quad \sum_k w_k = 1
$$

The web app uses $K = 2$. Each style can use a different brushstroke scale.

**3. Brushstroke scale**
The style image is resized (for example 128 px vs 1280 px) before its statistics are computed:
- **Small size (128):** finer, denser strokes.
- **Large size (1280):** broader, softer strokes.

Differences between scales were evaluated with Mean Squared Error (MSE).

### Speed
A single feed-forward pass: roughly 1.3–2 s on a desktop CPU in this implementation (measured), and much faster on a GPU, depending on resolution.

---

## ⚠️ Limitations and Tips

AdaIN matches only global feature statistics (channel mean and standard deviation) and
reconstructs the image from 1/8-resolution features, so very fine details can be lost.
Small faces, lips, eyes, text, and busy scenes with many small objects may be distorted.

Best results:
- Landscapes with simple, high-contrast subjects.
- Simple portraits with a plain or minimal background, with the face large in the frame.
- If details get lost, lower style strength (about 0.5–0.7), try Fine brushstrokes, or crop closer to the subject.

---

## 🛠️ Requirements

`torch`, `torchvision`, `streamlit`, `Pillow`, `numpy`, `psutil` (see `Deployment/requirements.txt`). The Gatys notebook additionally uses `matplotlib`.

---

## 📚 References and Credits

- Gatys, Ecker, Bethge. *A Neural Algorithm of Artistic Style* (2015).
- Huang, Belongie. *Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization* (2017).
- Pretrained AdaIN weights: [naoto0804/pytorch-AdaIN](https://github.com/naoto0804/pytorch-AdaIN).
- Built-in style images are public-domain artworks, sourced from Wikimedia Commons.
