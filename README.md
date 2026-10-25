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
🧠 Encoder–Decoder Pipeline

Utilizes a custom VGG-19 encoder (frozen up to relu4_1) paired with a reflection-padded, symmetric decoder trained to reconstruct pixel-space images from normalized feature representations.

🎨 Feature-Space Alignment

Instead of matching computationally expensive Gram matrices, Adaptive Instance Normalization (AdaIN) aligns the channel-wise mean (
𝜇
) and standard deviation (
𝜎
) of content features with those of style features.

For multi-style transfer, the combined feature representation is computed as:

𝐹
combined
=
∑
𝑘
=
1
𝐾
𝑤
𝑘
⋅
AdaIN
⁡
(
𝑓
(
𝑐
)
,
𝑓
(
𝑠
𝑘
)
)

where:

𝐹
combined
 represents the blended feature representation.
𝑤
𝑘
 denotes the weight assigned to the 
𝑘
-th style.
𝑓
(
𝑐
)
 represents the content image features.
𝑓
(
𝑠
𝑘
)
 represents the features extracted from the 
𝑘
-th style image.
𝐾
 is the total number of style images.
⚡ Real-Time Inference

Achieves real-time style transfer through a single feed-forward pass (~0.01 seconds), without retraining the network for different styles.

🎛️ Advanced Runtime Controls

The pipeline supports flexible runtime customization without modifying or retraining network weights.

1. Style Interpolation (
𝛼
) & Multi-Style Blending

Controls the stylization strength by interpolating between content features and stylized features:

𝐹
output
=
(
1
−
𝛼
)
𝑓
(
𝑐
)
+
𝛼
𝐹
stylized

where 
𝛼
∈
[
0
,
1
]
 controls the interpolation strength:

𝛼
=
0
: Preserves the original content representation.
𝛼
=
1
: Applies the full stylized representation.
0
<
𝛼
<
1
: Produces an intermediate blend.

Multiple artistic styles can also be combined using weighted feature-space blending, enabling compositions inspired by Katsushika Hokusai's The Great Wave off Kanagawa, Claude Monet's Water Lilies, and other style references.

𝐹
combined
=
∑
𝑘
=
1
𝐾
𝑤
𝑘
⋅
AdaIN
⁡
(
𝑓
(
𝑐
)
,
𝑓
(
𝑠
𝑘
)
)
2. Brushstroke Scale Control

Dynamically adjusts the style image resolution (e.g., 128 px vs. 1280 px) to influence the apparent scale of transferred textures and brushstrokes.

Lower resolution: Encourages broader, coarser stylistic patterns.
Higher resolution: Can preserve finer textures and more detailed stylistic patterns.

The resulting structural differences can be evaluated using Mean Squared Error (MSE) analysis.
