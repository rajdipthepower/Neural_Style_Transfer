# DeepArt Experiment: Neural Style Transfer in PyTorch

A repository documenting the implementation and hyperparameter optimization process for Neural Style Transfer using PyTorch. This project blends the content of a target image with the visual style of a reference image through iterative optimization.

---

## 🔬 Experiment Workflow & Hyperparameter Tuning

Training a neural style transfer model requires balancing content preservation and style replication. Below is the iterative tuning process used to achieve the final output:

1. **Initial Attempt (Failed Initialization)**
   * **Configuration**: 500 epochs, `content_weight = 1`, `style_weight = 5e-6`, initialized with the **original target image**.
   * **Result**: The optimization failed to converge effectively because starting directly from the content image trapped the network in a local minimum, producing minimal stylistic change.

2. **Second Attempt (Over-Stylization / Noise)**
   * **Configuration**: 700 epochs, `content_weight = 1`, `style_weight = 5e-6`, initialized with **random noise**.
   * **Result**: While the higher epoch count allowed more iteration, the low content weight relative to the style weight caused the structural integrity of the image to collapse, rendering the output entirely meaningless and noisy.

3. **Final Successful Render (Optimized Balance)**
   * **Configuration**: Optimized epochs, `content_weight = 200`, `style_weight = 1e-8`.
   * **Result**: Significantly boosting the content weight and scaling down the style weight forced the algorithm to strictly preserve the underlying shapes and structures of the original photograph while cleanly mapping the color and texture patterns of the style image.

---

## 📈 Loss Curve & Training Observations

During the optimization process, tracking the loss data revealed key insights about convergence:
* **Content Loss**: Starts very low when using initial target seeding, but fluctuates and spikes early on under random initialization as the model attempts to build recognizable structures from static noise.
* **Style Loss**: Decreases rapidly within the first 100–200 epochs as high-frequency textures and color distributions are quickly transferred from the style reference.
* **Curve Saturation**: Beyond a certain threshold (typically past the mid-point of training), both losses begin to flatten out asymptotically. Continuing training past saturation yields diminishing returns, primarily resulting in microscopic pixel-level jitter rather than structural improvement.

---

## 🛠️ Prerequisites & Dependencies
Ensure you have the following installed:
* `torch`
* `torchvision`
* `Pillow` (PIL)
* `numpy`
* `matplotlib`
* `pathlib`

---

## 🚀 Extension: Real-Time Arbitrary Style Transfer (AdaIN)
While Gatys' optimization produces stunning micro-textures, it is computationally expensive (taking minutes per image via pixel-level optimization). To bridge the gap between artistic quality and speed, this project has been extended to implement **Adaptive Instance Normalization (AdaIN)** in PyTorch.

### 🧠 Core Architecture & Mechanism
- **Encoder-Decoder Pipeline**: Utilizes a custom VGG-19 encoder (frozen up to `relu4_1`) paired with a reflection-padded symmetric decoder trained to invert feature statistics back into pixel space.
- **Feature-Space Alignment**: Instead of matching slow Gram matrices, AdaIN directly aligns the channel-wise mean ($\mu$) and standard deviation ($\sigma$) of the content features with the style features:
  $$ \text{AdaIN}(x, y) = \sigma(y) \left( \frac{x - \mu(x)}{\sigma(x)} \right) + \mu(y) $$
- **Efficiency**: Achieves real-time style transfer in a single feed-forward pass (~0.01 seconds) without network retraining.

### 🎛️ Advanced Runtime Controls
Without modifying or retraining network weights, the pipeline supports four flexible runtime features:
1. **Style Interpolation ($\alpha$ & Multi-Style Blending)**: Smoothly blends between content/style reconstructions or combines multiple art styles (e.g., Human Face + Katsushika Hokusai’s The Great Wave off Kanagawa + Claude Monet's Water Lilies) via linear feature-space weighting:
   $$ \text{Combined Feature} = \sum w_k \cdot \text{AdaIN}(f(c), f(s_k)) $$
2. **Brushstroke Scale Control**: Dynamically scales the style input resolution (e.g., $128 \text{px}$ vs. $1280 \text{px}$) to shift VGG's receptive fields between chunky macro-strokes and fine micro-textures (verified via MSE structural analysis).
