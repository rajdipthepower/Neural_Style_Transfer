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
# Quick preview of the content image
with Image.open(content) as img:
    display(img.resize((500, 500)))
