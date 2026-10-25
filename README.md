
# DeepArt Experiment: Neural Style Transfer in PyTorch

A repository documenting the implementation and hyperparameter optimization process for Neural Style Transfer using PyTorch. This project blends the content of a target image with the visual style of a reference image through iterative optimization.

---

## 🔬 Experiment Workflow & Hyperparameter Tuning

Training a neural style transfer model requires balancing content preservation and style replication. Below is the iterative tuning process used to achieve the final output:

1. **Initial Attempt (Failed Initialization)**
   - **Configuration**: 500 epochs, `content_weight = 1`, `style_weight = 5e-6`, initialized with the **original target image**.
   - **Result**: The optimization failed to converge effectively because starting directly from the content image trapped the optimization in a local minimum, producing minimal stylistic change.

2. **Second Attempt (Over-Stylization / Noise)**
   - **Configuration**: 700 epochs, `content_weight = 1`, `style_weight = 5e-6`, initialized with **random noise**.
   - **Result**: While the higher epoch count allowed more iterations, the optimization produced noisy, structurally incoherent outputs due to an insufficient balance between content preservation and style transfer.

3. **Final Successful Render (Optimized Balance)**
   - **Configuration**: Optimized epochs, `content_weight = 200`, `style_weight = 1e-8`.
   - **Result**: Increasing the content weight and reducing the style weight encouraged the algorithm to preserve the underlying shapes and structures of the original photograph while transferring color and texture patterns from the style image.

---

## 📈 Loss Curve & Training Observations

During the optimization process, tracking the loss data revealed key insights about convergence:

- **Content Loss**: Starts very low when initialized with the target image, but can fluctuate and spike early under random initialization as the optimization attempts to reconstruct recognizable structures from noise.
- **Style Loss**: Decreases rapidly within the first 100–200 epochs as textures and color distributions are transferred from the style reference.
- **Curve Saturation**: Beyond a certain point, both losses begin to flatten asymptotically. Continuing training after saturation yields diminishing returns, potentially introducing minor pixel-level changes without significant structural improvement.

---

## 🛠️ Prerequisites & Dependencies

Ensure you have the following installed:

- `torch`
- `torchvision`
- `Pillow`
- `numpy`
- `matplotlib`
- `pathlib`

---

## 🚀 Extension: Real-Time Arbitrary Style Transfer (AdaIN)

While Gatys' optimization-based approach can produce detailed textures, it is computationally expensive because it iteratively optimizes image pixels. To bridge the gap between artistic quality and speed, this project has been extended to implement **Adaptive Instance Normalization (AdaIN)** in PyTorch.

### 🧠 Core Architecture & Mechanism

#### Encoder–Decoder Pipeline

Utilizes a custom VGG-19 encoder (frozen up to `relu4_1`) paired with a reflection-padded, symmetric decoder trained to reconstruct pixel-space images from normalized feature representations.

#### 🎨 Feature-Space Alignment

Instead of matching computationally expensive Gram matrices, **Adaptive Instance Normalization (AdaIN)** aligns the channel-wise mean ($\mu$) and standard deviation ($\sigma$) of content features with those of style features.

The AdaIN operation is defined as:

$$
\operatorname{AdaIN}(f(c), f(s)) =
\sigma(f(s))
\left(
\frac{f(c)-\mu(f(c))}
{\sigma(f(c))}
\right)
+\mu(f(s))
$$

For multi-style transfer, the combined feature representation is computed as:

$$
F_{\text{combined}} =
\sum_{k=1}^{K}
w_k \cdot
\operatorname{AdaIN}\bigl(f(c), f(s_k)\bigr)
$$

Where:

- $F_{\text{combined}}$ represents the blended feature representation.
- $w_k$ denotes the weight assigned to the $k$-th style.
- $f(c)$ represents the content image features.
- $f(s_k)$ represents the features extracted from the $k$-th style image.
- $K$ is the total number of style images.

#### ⚡ Real-Time Inference

Enables style transfer through a single feed-forward pass without retraining the network for every new style. Inference time can be approximately **0.01 seconds**, depending on hardware, input resolution, and implementation.

---


## 🎛️ Advanced Runtime Controls

The pipeline supports flexible runtime customization without modifying or retraining network weights.

### 1. Style Interpolation ($\alpha$) & Multi-Style Blending

Controls the stylization strength by interpolating between the content features and the stylized features:

$$
F_{\text{output}} = (1-\alpha)f(c) + \alpha F_{\text{stylized}}
$$

Where $\alpha \in [0,1]$ controls the interpolation strength:

- $\alpha = 0$: Preserves the original content representation.
- $\alpha = 1$: Applies the full stylized representation.
- $0 < \alpha < 1$: Produces an intermediate blend.

Multiple artistic styles can also be combined using weighted feature-space blending, enabling compositions inspired by Katsushika Hokusai's *The Great Wave off Kanagawa*, Claude Monet's *Water Lilies*, and other artistic references.

$$
F_{\text{combined}} =
\sum_{k=1}^{K}
w_k \cdot
\text{AdaIN}\bigl(f(c), f(s_k)\bigr)
$$

### 2. Brushstroke Scale Control

Dynamically adjusts the style image resolution (e.g., **128 px vs. 1280 px**) to influence the apparent scale of transferred textures and brushstrokes.

- **Lower resolution:** Can encourage broader, coarser stylistic patterns.
- **Higher resolution:** Can preserve finer textures and more detailed stylistic patterns.

The resulting structural differences are evaluated using **Mean Squared Error (MSE)** analysis.
