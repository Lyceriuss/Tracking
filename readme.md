# Real-Time Pedestrian Attribute Recognition (PAR) & Tracking Pipeline

A high-performance computer vision system combining **YOLO (ByteTrack)** for multi-object tracking with a custom-engineered lightweight CNN (**Besten**) for real-time pedestrian attribute classification in forensic and security environments.

---

##  About the Model (`Besten`)

Unlike traditional heavy computer vision backbones (such as ResNet-50) that can suffer from diminishing returns and bloated parameter counts, this pipeline is built around **`Besten`** my custom-engineered, lightweight CNN specifically optimized for cost-effective performance. You can customize the layers to get whatever depth and power you need to maximize feauture extraction, then you can prune it down, keeping as much as you need. Besten can be many things however and it works even better as a Hydra with several diffrerent heads that have different SE and group_width.

### Architectural Advantages
1. **Grouped Convolutions (RegNet Efficiency):** Inside the bottleneck layers (`conv2`), the model uses grouped convolutions (`groups = mid_c // group_width`), drastically slashing Floating Point Operations (FLOPs) and parameter size while preserving high-dimensional representation power.
2. **Squeeze-and-Excitation (SE) Attention:** Every bottleneck integrates an SE channel attention block (`nn.AdaptiveAvgPool2d` + reduction/expansion layers + hard-sigmoid activation) that dynamically recalibrates channel-wise feature responses, allowing the network to emphasize salient pedestrian traits (e.g., backpacks, logos) and suppress background noise.
3. **Custom `RC_Layer` (Residual Calibration):** Applied in the final high-level blocks (`out_c >= 256`), the specialized `RC_Layer` uses learnable parameters ($\mu$ and $\sigma$) alongside a fast error-function approximation (`fast_erf`) to center, scale, and soft-threshold features. This acts as an internal regularizer, stabilizing semantic representations for sparse or tricky attributes.
4. **Flexible & Scalable Layout:** Rather than a rigid hardcoded architecture like standard ResNet variants, `BestenSingle` uses a dynamic `layer_config` tuple list `[(repeats, out_channels, stride), ...]`, making it uniquely adaptable for hardware-constrained edge deployments.

###  Hybrid Design Philosophy (Bridging CNNs & Vision Transformers)
While structurally a CNN, `BestenSingle` incorporates modern design paradigms shared with Vision Transformers (ViTs):
* **Channel-Wise Attention:** Much like a Transformer's self-attention mechanism weights different tokens, the SE blocks dynamically score and scale feature channels based on global context, ensuring the model focuses on critical semantic clues rather than static spatial templates.
* **Global Context Flow:** Through adaptive pooling, global image context is captured and redistributed across feature maps early in the pipeline, mimicking the wide receptive fields characteristic of attention-based models.

---

## 📈 Optimization & Production Readiness
* **Data-Driven Dynamic Thresholding:** Bypassed naive 0.5 decision boundaries by performing grid-search validation across 99 thresholds per class, optimizing for Mean Accuracy (mA) and driving overall performance into the high **88% mA** bracket.
* **Model Pruning:** Utilizing **L1 Unstructured Magnitude Pruning**, the model was compressed down to **49.67% sparsity** (reducing active parameters to **3.2M**) with **0.0% loss in validation accuracy**.
* **Storage Footprint:** Raw PyTorch checkpoint size reduced to 25.18 MB, collapsing down to **14.27 MB** compressed—ideal for rapid OTA updates to edge cameras or smart-city infrastructure control centers.
---

## 📊 Attribute Schema (36 Classes)
* **Viewpoint:** Front, Back, Side
* **Demographics & Identity:** Female, Male, Age_Child, Age_Adult, Age_Senior
* **Hair & Head:** Bald, Short_Hair, Long_Hair, Hat, Glasses, Muffler_Scarf
* **Upper Body:** ShortSleeve, LongSleeve, Tshirt, Jacket, LongCoat, Logo, Plaid, Stripe
* **Lower Body:** Trousers, Jeans, Shorts, Skirt_or_Dress
* **Footwear:** Boots, Sneakers, LeatherShoes, Sandals
* **Bags & Carrying:** Backpack, ShoulderBag, HandBag, PlasticBag, CarryingOther

---

## 🛠️ Training Augmentation Pipeline
To ensure robustness against occlusion, poor lighting, and motion blur in real-world camera feeds, our training regimen incorporates rigorous spatial, color, and occlusion augmentations:

```python
T.Compose([
    T.RandomHorizontalFlip(p=0.5), 
    T.RandomAffine(degrees=5, translate=(0.05, 0.05)),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    T.RandomApply([T.GaussianBlur(kernel_size=3)], p=0.2),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    T.RandomErasing(p=0.2, scale=(0.02, 0.1)) 
])