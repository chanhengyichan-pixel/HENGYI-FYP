# 🦺 Edge AI-Based PPE Compliance Monitoring System

> Final Year Project — Multimedia University (MMU)  
> Faculty of Information Science and Technology (FIST) | 2026

---

## 📌 What is this project?

This project compares **4 AI object detection models** to find the best one for automatically detecting whether workers are wearing Personal Protective Equipment (PPE) — such as safety helmets and vests — in industrial environments like construction sites and factories.

---

## 🏆 Final Result

| Rank | Model | Avg mAP@0.5 | Avg Recall | Avg FPS |
|------|-------|-------------|------------|---------|
| 🥇 1st | **YOLOv8** | 78.63% | 74.93% | 78.7 |
| 🥈 2nd | YOLOv11 | 77.42% | 73.43% | 81.2 |
| 🥉 3rd | YOLOv9 | 78.01% | 73.85% | 62.7 |
| 4th | SSD MobileNet V2 | 35.13% | 42.91% | 50.5 |

✅ **Recommended Model: YOLOv8**

---

## 🗂️ Datasets Used

| Dataset | Industry | Images | Classes |
|---------|----------|--------|---------|
| SH17 | Manufacturing | 8,099 | 17 |
| SFCHD | Chemical Plant | 12,373 | 7 |
| CHVG | Construction | 1,699 | 8 |
| SHEL5K | Construction | ~5,000 | 6 |
| Roboflow Construction Safety | Construction | 1,206 | 5 |

> ⚠️ Datasets are not included in this repo due to size (~5GB).

---

## 🧪 Experiments

- **Planned:** 25 experiments (5 models × 5 datasets)
- **Completed:** 20 experiments
- **Excluded:** Faster R-CNN — GPU memory too small (RTX 4050, 6GB VRAM)

---

## 📊 Metrics Used

| Type | Metrics |
|------|---------|
| Accuracy | mAP@0.5, Precision, Recall, F1-Score |
| Efficiency | FPS, Latency (ms) |

**Recall weighted highest (30%)** — missing a violation is more dangerous than a false alarm.

---

## ⚙️ Hardware & Software

| | |
|--|--|
| GPU | NVIDIA RTX 4050 Laptop (6GB VRAM) |
| CPU | AMD Ryzen 7 7445HS |
| RAM | 16GB |
| OS | Linux Mint 22.2 |
| Python | 3.12.3 |
| PyTorch | 2.11.0 (CUDA 12.8) |
| Ultralytics | 8.4.51 |

---

## 👨‍💻 Author

**Chan Heng Yi**  
Multimedia University (MMU), Malaysia  
Faculty of Information Science and Technology (FIST)
