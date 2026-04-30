# 🚦 TrafficSignNet — CNN Architecture Comparison

A systematic benchmarking study of three CNN architectures for traffic sign classification on the [GTSRB dataset](https://benchmark.ini.rub.de/gtsrb_news.html), evaluating accuracy, generalisation, inference speed, and model size for autonomous driving deployment scenarios.

---

## 📋 Project Overview

This project trains and evaluates three architectures on the German Traffic Sign Recognition Benchmark (GTSRB) — a 43-class dataset of ~50,000 real-world traffic sign images — and produces a full comparison study across multiple metrics.

| Model | Test Accuracy | Macro F1 | FPS | Model Size |
|---|---|---|---|---|
| Lightweight CNN (scratch) | 0.9008 | 0.7806 | 1064.9 | 9.16 MB |
| ResNet-18 (transfer learning) | 0.9599 | 0.9096 | 429.6 | 42.80 MB |
| EfficientNet-B0 (transfer learning) | 0.9508 | 0.9004 | 134.3 | 15.79 MB |

---

## 🔑 Key Findings

- **ResNet-18** achieves the highest test accuracy (96%) and Macro F1 (0.91), making it the best choice for server-side deployment where accuracy is the priority.
- **EfficientNet-B0** comes within 1% of ResNet-18's accuracy at less than half the model size (16MB vs 43MB) — optimal for edge device deployment.
- **Lightweight CNN** delivers extraordinary inference speed (1064 FPS) with acceptable accuracy (90%), suitable for extreme real-time constraints.
- The gap between CNN accuracy (0.90) and F1 (0.78) reveals that class imbalance affects lightweight models significantly more than pretrained architectures.
- EfficientNet's depthwise separable convolutions are slower on T4 GPU (134 FPS) than ResNet-18 (429 FPS) — its efficiency advantage is realised on edge CPUs, not datacenter GPUs.

---

## 🗂️ Dataset

**GTSRB — German Traffic Sign Recognition Benchmark**

- 43 traffic sign classes
- ~39,000 training images / ~12,000 test images
- Variable lighting, blur, occlusion — real-world conditions
- Loaded directly via `torchvision.datasets.GTSRB`

---

## 🏗️ Architecture Details

### Lightweight CNN (Baseline)
- Built from scratch using stacked `Conv2d → BatchNorm → ReLU` blocks
- 3 convolutional stages with MaxPooling and Dropout
- Input: 64×64 | Parameters: ~2–3M
- Trained end-to-end with GTSRB-specific normalization

### ResNet-18 (Transfer Learning)
- Pretrained on ImageNet (1.2M images, 1000 classes)
- Final `fc` layer replaced: `Linear(512 → 43)`
- Two-phase training: freeze backbone (5 epochs) → full fine-tune (15 epochs)
- Input: 224×224 | Parameters: ~11M

### EfficientNet-B0 (Transfer Learning)
- Pretrained on ImageNet via compound scaling (depth + width + resolution)
- Final classifier layer replaced: `Linear(1280 → 43)`
- Same two-phase training strategy as ResNet-18
- Input: 224×224 | Parameters: ~5.3M

---

## 🔁 Training Strategy

All pretrained models follow a two-phase training approach to prevent catastrophic forgetting:

```
Phase 1 (5 epochs)  — Freeze backbone, train head only (lr=1e-3)
Phase 2 (15 epochs) — Unfreeze all layers, fine-tune (lr=1e-4)
```

- **Loss:** CrossEntropyLoss
- **Optimizer:** Adam
- **Scheduler:** StepLR
- **Augmentation (train only):** RandomHorizontalFlip, ColorJitter, RandomRotation

---

## 📊 Evaluation Metrics

- **Test Accuracy** — top-1 classification accuracy on held-out test set
- **Macro F1** — unweighted F1 across all 43 classes (handles class imbalance)
- **Inference Speed** — single-image latency (ms) and throughput (FPS) on T4 GPU
- **Model Size** — `.pth` checkpoint size on disk (MB)
- **Confusion Matrix** — per-class failure analysis
- **Per-class Report** — precision, recall, F1 for all 43 classes

---

## 📁 Repository Structure

```
gtsrb-architecture-comparison/
│
├── notebook.ipynb          # Full training + evaluation notebook
├── README.md
│
├── results/
│   ├── comparison_results.csv
│   ├── comparison_charts.png
│   ├── confusion_matrix_Lightweight_CNN.png
│   ├── confusion_matrix_ResNet18.png
│   ├── confusion_matrix_EfficientNet_B0.png
│   ├── lightweight_cnn_curves.png
│   ├── ResNet-18_curves.png
│   └── EfficientNet-B0_curves.png
│
└── weights/
    ├── lightweight_cnn_best.pth
    ├── resnet18_best.pth
    └── effnet_best.pth
```

---

## ⚙️ Setup & Usage

### Requirements

```bash
pip install torch torchvision scikit-learn matplotlib seaborn pandas
```

### Run the Notebook

The notebook is self-contained. Dataset downloads automatically on first run via `torchvision.datasets.GTSRB`.

```python
# All configuration is centralized
CONFIG = {
    "batch_size"     : 64,
    "epochs"         : 20,
    "lr"             : 1e-3,
    "img_size_small" : 64,
    "img_size_large" : 224,
    "num_classes"    : 43,
    "seed"           : 42,
    "device"         : "cuda" if torch.cuda.is_available() else "cpu"
}
```

Recommended environment: **Google Colab** or **Kaggle** with T4 GPU.

---

## 🚗 Deployment Recommendation

| Scenario | Recommended Model | Reason |
|---|---|---|
| Server-side (cloud inference) | ResNet-18 | Highest accuracy + F1 |
| Edge device (Raspberry Pi, Jetson) | EfficientNet-B0 | 3x smaller, competitive accuracy |
| Extreme real-time constraint | Lightweight CNN | 1064 FPS, 9MB footprint |

---

## 🛠️ Built With

- [PyTorch](https://pytorch.org/) & [Torchvision](https://pytorch.org/vision/)
- [scikit-learn](https://scikit-learn.org/) — metrics
- [Matplotlib](https://matplotlib.org/) & [Seaborn](https://seaborn.pydata.org/) — visualizations
- [GTSRB Dataset](https://benchmark.ini.rub.de/gtsrb_news.html)

---

## 👤 Author

**MJay**  
AI/Computer Vision Researcher | Virtual University of Pakistan  
Focus: Autonomous Driving Perception, Traffic Sign Detection

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
