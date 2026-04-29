# TrafficDetectPK 🚗🛺🏍️

Fine-tuning YOLOv8 for Pakistani traffic object detection — bridging the domain gap between Western benchmark datasets and real South Asian road conditions.

---

## Project Overview

Standard object detection models are trained predominantly on Western traffic datasets (COCO, ImageNet) and fail to generalize on South Asian road environments — which include unique vehicle classes like Auto Rickshaws, Chingchi, and Motorcycles operating in dense, unstructured traffic.

This project fine-tunes **YOLOv8n** on a Pakistan-specific traffic dataset and evaluates its performance on real local road footage, with an honest analysis of where the model succeeds and where it fails.

---

## Dataset

- **Source:** [Traffic-Erawl Dataset](https://universe.roboflow.com/moiz-chauhan-u4zyj/traffic-erawl/dataset/6) via Roboflow Universe
- **License:** CC BY 4.0
- **Classes:** 8
- **Format:** YOLOv8

| # | Class |
|---|-------|
| 0 | Auto Rickshaw |
| 1 | Car-Jeeps-Vans-Taxi |
| 2 | Cart-Chingchi |
| 3 | Large Buses |
| 4 | Mini Buses |
| 5 | Motorcycles-Scooters |
| 6 | Pickups |
| 7 | Trucks and Trailers |

---

## Model

| Setting | Value |
|---|---|
| Architecture | YOLOv8n (nano) |
| Pretrained Weights | ImageNet |
| Epochs | 50 |
| Image Size | 640px |
| Batch Size | 16 |
| Training Platform | Google Colab (T4 GPU) |

---

## Results

| Metric | Score |
|---|---|
| mAP50 | 0.787 |
| mAP50-95 | 0.588 |
| Precision | 0.737 |
| Recall | 0.748 |

---

## Inference on Local Road Footage

The trained model was tested on self-recorded footage from a local road in Sargodha, Punjab, Pakistan.

### Sample Detections

<p align="center">
  <img src=TrafficDetectPK/sample_images/Frame_01.PNG" width="30%"/>
  <img src="sample_images/Frame_02.png" width="30%"/>
  <img src="sample_images/Frame_03.png" width="30%"/>
</p>

---

## Failure Analysis

While the model performed well on the benchmark dataset (mAP50: 0.787), inference on local road footage revealed clear limitations:

- **Class bias:** Model over-predicts `Pickup` and `Cart-Chingchi` classes on unseen footage
- **Domain shift:** Training images differ in camera angle, resolution, and lighting compared to self-recorded footage
- **Class imbalance:** Underrepresented classes (Large Buses, Mini Buses) show weaker recall on real footage

These findings highlight a fundamental challenge in autonomous driving perception — models trained on existing datasets do not generalize reliably to new geographic and environmental contexts without domain adaptation.

### Future Work
- Collect and annotate local road images to retrain with diverse data
- Apply class rebalancing / oversampling for underrepresented classes
- Explore domain adaptation techniques to bridge the distribution gap

---

## How to Run

**1. Clone the repo**
```bash
git clone https://github.com/Jahanzaib-56/Deep-Learning-Computer-Vision.git
cd Deep-Learning-Computer-Vision/TrafficDetectPK
```

**2. Install dependencies**
```bash
pip install ultralytics roboflow python-dotenv
```

**3. Download dataset**

Create a `.env` file with your Roboflow API key:

Then run:
```bash
python download_dataset.py
```

**4. Train**
```bash
python train.py
```

**5. Run inference**
```bash
python inference.py
```

---

## Tech Stack

- YOLOv8 (Ultralytics)
- PyTorch
- Roboflow
- OpenCV
- Google Colab (T4 GPU)

---

## Acknowledgements

Dataset by [Moiz Chauhan](https://universe.roboflow.com/moiz-chauhan-u4zyj) on Roboflow Universe, licensed under CC BY 4.0.
