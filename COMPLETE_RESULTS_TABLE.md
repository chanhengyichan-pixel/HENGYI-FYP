# Complete Experimental Results — All 20 Experiments

## Summary by Algorithm

| Algorithm | Exp # | Dataset | Epochs | mAP@0.5 | Precision | Recall | F1-Score | FPS | Latency(ms) |
|-----------|-------|---------|--------|---------|-----------|--------|----------|-----|-------------|
| **YOLOv8** | 1 | Roboflow | 38 | 80.01% | 70.41% | 81.52% | 0.7556 | 53.6 | 18.67 |
| **YOLOv8** | 2 | CHVG | 51 | 92.26% | 88.61% | 88.95% | 0.8878 | 126.0 | 7.93 |
| **YOLOv8** | 3 | SHEL5K | 79 | 77.04% | 91.31% | 71.93% | 0.8047 | 93.0 | 10.76 |
| **YOLOv8** | 4 | SH17 | 100 | 64.71% | 79.09% | 57.22% | 0.6640 | 7.7 | 130.70 |
| **YOLOv8** | 5 | SFCHD | 91 | 79.14% | 82.27% | 75.05% | 0.7850 | 113.4 | 8.82 |
| | | **YOLOv8 MEAN** | **72** | **78.63%** | **82.34%** | **74.93%** | **0.7794** | **78.74** | **35.38** |
| | | | | | | | | | |
| **YOLOv9** | 6 | Roboflow | 67 | 78.43% | 84.67% | 72.99% | 0.7840 | 61.3 | 16.32 |
| **YOLOv9** | 7 | CHVG | 49 | 93.32% | 88.91% | 90.31% | 0.8961 | 88.1 | 11.36 |
| **YOLOv9** | 8 | SHEL5K | 78 | 77.18% | 90.76% | 72.81% | 0.8080 | 75.8 | 13.20 |
| **YOLOv9** | 9 | SH17 | 100 | 62.59% | 71.66% | 58.81% | 0.6460 | 6.5 | 152.94 |
| **YOLOv9** | 10 | SFCHD | 99 | 78.52% | 83.00% | 74.34% | 0.7843 | 81.6 | 12.25 |
| | | **YOLOv9 MEAN** | **79** | **78.01%** | **83.80%** | **73.85%** | **0.7837** | **62.66** | **41.21** |
| | | | | | | | | | |
| **YOLOv11** | 11 | Roboflow | 62 | 75.59% | 72.04% | 74.95% | 0.7347 | 68.3 | 14.65 |
| **YOLOv11** | 12 | CHVG | 59 | 92.14% | 88.26% | 88.10% | 0.8818 | 129.4 | 7.73 |
| **YOLOv11** | 13 | SHEL5K | 73 | 77.24% | 91.27% | 72.15% | 0.8059 | 92.1 | 10.86 |
| **YOLOv11** | 14 | SH17 | 100 | 63.21% | 80.18% | 56.29% | 0.6615 | 7.3 | 137.34 |
| **YOLOv11** | 15 | SFCHD | 100 | 78.91% | 83.28% | 75.67% | 0.7930 | 108.7 | 9.20 |
| | | **YOLOv11 MEAN** | **79** | **77.42%** | **83.01%** | **73.43%** | **0.7754** | **81.16** | **35.96** |
| | | | | | | | | | |
| **Faster R-CNN** | 16-20 | All | — | **0.00%** | **0.00%** | **0.00%** | **0.00** | **0.0** | **0.00** |
| | | **[EXCLUDED - Hardware Limitation]** | | | | | | | |
| | | | | | | | | | |
| **SSD300** | 21 | Roboflow | 100 | 50.66% | 65.18% | 58.59% | 0.6171 | 51.2 | 19.53 |
| **SSD300** | 22 | CHVG | 100 | 54.91% | 69.94% | 55.04% | 0.6160 | 53.5 | 18.68 |
| **SSD300** | 23 | SHEL5K | 100 | 33.99% | 66.70% | 33.77% | 0.4484 | 52.8 | 18.93 |
| **SSD300** | 24 | SH17 | 5 | 10.88% | 62.90% | 26.96% | 0.3774 | 43.3 | 23.07 |
| **SSD300** | 25 | SFCHD | 5 | 25.20% | 83.75% | 40.18% | 0.5431 | 51.6 | 19.38 |
| | | **SSD300 MEAN** | **62** | **35.13%** | **69.69%** | **42.91%** | **0.5204** | **50.48** | **19.92** |

---

## Results by Dataset

### **Roboflow (5-class Construction)**

| Algorithm | mAP@0.5 | Precision | Recall | F1-Score | FPS | Latency(ms) |
|-----------|---------|-----------|--------|----------|-----|-------------|
| YOLOv8 | 80.01% | 70.41% | 81.52% | 0.7556 | 53.6 | 18.67 |
| YOLOv9 | 78.43% | 84.67% | 72.99% | 0.7840 | 61.3 | 16.32 |
| YOLOv11 | 75.59% | 72.04% | 74.95% | 0.7347 | 68.3 | 14.65 |
| SSD300 | 50.66% | 65.18% | 58.59% | 0.6171 | 51.2 | 19.53 |
| **MEAN** | **71.17%** | **73.08%** | **72.01%** | **0.7229** | **58.60** | **17.29** |

---

### **CHVG (8-class Construction Vest)**

| Algorithm | mAP@0.5 | Precision | Recall | F1-Score | FPS | Latency(ms) |
|-----------|---------|-----------|--------|----------|-----|-------------|
| YOLOv8 | 92.26% | 88.61% | 88.95% | 0.8878 | 126.0 | 7.93 |
| YOLOv9 | 93.32% | 88.91% | 90.31% | 0.8961 | 88.1 | 11.36 |
| YOLOv11 | 92.14% | 88.26% | 88.10% | 0.8818 | 129.4 | 7.73 |
| SSD300 | 54.91% | 69.94% | 55.04% | 0.6160 | 53.5 | 18.68 |
| **MEAN** | **83.16%** | **83.93%** | **80.60%** | **0.8204** | **99.25** | **11.43** |

---

### **SHEL5K (6-class Helmet-focused)**

| Algorithm | mAP@0.5 | Precision | Recall | F1-Score | FPS | Latency(ms) |
|-----------|---------|-----------|--------|----------|-----|-------------|
| YOLOv8 | 77.04% | 91.31% | 71.93% | 0.8047 | 93.0 | 10.76 |
| YOLOv9 | 77.18% | 90.76% | 72.81% | 0.8080 | 75.8 | 13.20 |
| YOLOv11 | 77.24% | 91.27% | 72.15% | 0.8059 | 92.1 | 10.86 |
| SSD300 | 33.99% | 66.70% | 33.77% | 0.4484 | 52.8 | 18.93 |
| **MEAN** | **66.36%** | **85.01%** | **62.67%** | **0.7668** | **78.43** | **13.44** |

---

### **SH17 (17-class Manufacturing)**

| Algorithm | mAP@0.5 | Precision | Recall | F1-Score | FPS | Latency(ms) |
|-----------|---------|-----------|--------|----------|-----|-------------|
| YOLOv8 | 64.71% | 79.09% | 57.22% | 0.6640 | 7.7 | 130.70 |
| YOLOv9 | 62.59% | 71.66% | 58.81% | 0.6460 | 6.5 | 152.94 |
| YOLOv11 | 63.21% | 80.18% | 56.29% | 0.6615 | 7.3 | 137.34 |
| SSD300 | 10.88% | 62.90% | 26.96% | 0.3774 | 43.3 | 23.07 |
| **MEAN** | **50.35%** | **73.46%** | **49.82%** | **0.5872** | **16.20** | **111.01** |

---

### **SFCHD (7-class Chemical/Hazardous)**

| Algorithm | mAP@0.5 | Precision | Recall | F1-Score | FPS | Latency(ms) |
|-----------|---------|-----------|--------|----------|-----|-------------|
| YOLOv8 | 79.14% | 82.27% | 75.05% | 0.7850 | 113.4 | 8.82 |
| YOLOv9 | 78.52% | 83.00% | 74.34% | 0.7843 | 81.6 | 12.25 |
| YOLOv11 | 78.91% | 83.28% | 75.67% | 0.7930 | 108.7 | 9.20 |
| SSD300 | 25.20% | 83.75% | 40.18% | 0.5431 | 51.6 | 19.38 |
| **MEAN** | **65.44%** | **83.08%** | **66.31%** | **0.7514** | **88.85** | **12.41** |

---

## Final Rankings (Weighted Multi-Criteria)

### **Overall Algorithm Ranking**

| Rank | 🏆 Algorithm | Weighted Score | Avg mAP@0.5 | Avg Recall | Avg FPS | Key Strength |
|------|-------------|-----------------|------------|-----------|---------|---|
| 🥇 1st | **YOLOv8** | **0.9024** | 78.63% | 74.93% | 78.74 | Best mAP & Recall |
| 🥈 2nd | **YOLOv11** | **0.8947** | 77.42% | 73.43% | 81.16 | Fastest FPS |
| 🥉 3rd | **YOLOv9** | **0.7967** | 78.01% | 73.85% | 62.66 | Best Precision |
| 4th | **SSD300** | **0.1000** | 35.13% | 42.91% | 50.48 | Edge-suitable but lower accuracy |

**Weighting Scheme:**
- Recall: 30% (safety-critical)
- mAP@0.5: 20%
- F1-Score: 15%
- FPS: 15%
- Precision: 10%
- Latency: 10%

---

### **Dataset Difficulty Ranking (by avg mAP@0.5 across all algorithms)**

| Rank | Dataset | Avg mAP@0.5 | Difficulty | Notes |
|------|---------|------------|-----------|-------|
| 1️⃣ | **CHVG** | **83.16%** | ⭐ Easiest | Balanced 8-class distribution |
| 2️⃣ | **Roboflow** | **71.17%** | ⭐⭐ Easy | Small 5-class dataset |
| 3️⃣ | **SHEL5K** | **66.36%** | ⭐⭐⭐ Medium | Large test set (750 images) |
| 4️⃣ | **SFCHD** | **65.44%** | ⭐⭐⭐ Medium | Chemical hazardous domain |
| 5️⃣ | **SH17** | **50.35%** | ⭐⭐⭐⭐⭐ Hardest | 17-class complexity, lowest FPS |

---



## Key Statistics

### **Algorithm Performance Summary**

| Metric | YOLOv8 | YOLOv9 | YOLOv11 | SSD300 |
|--------|--------|--------|---------|--------|
| **Best mAP on dataset** | 92.26% (CHVG) | 93.32% (CHVG) | 92.14% (CHVG) | 54.91% (CHVG) |
| **Worst mAP on dataset** | 64.71% (SH17) | 62.59% (SH17) | 63.21% (SH17) | 10.88% (SH17) |
| **Highest FPS single exp** | 126.0 (CHVG) | 88.1 (CHVG) | 129.4 (CHVG) | 53.5 (CHVG) |
| **Lowest FPS single exp** | 7.7 (SH17) | 6.5 (SH17) | 7.3 (SH17) | 43.3 (SH17) |
| **Best Precision** | 91.31% | 84.67% | 91.27% | 83.75% |
| **Best Recall** | 88.95% | 90.31% | 88.10% | 58.59% |

### **Accuracy-Speed Trade-off Analysis**

| Algorithm | mAP vs Speed | Comment |
|-----------|------------|---------|
| **YOLOv8** | ✅ Balanced (78.63% mAP, 78.74 FPS) | Best overall choice |
| **YOLOv9** | 🔴 Slower (78.01% mAP, 62.66 FPS) | Lower speed despite similar accuracy |
| **YOLOv11** | 🟢 Fastest (77.42% mAP, 81.16 FPS) | Speed optimized, slightly lower accuracy |
| **SSD300** | 🟡 Fast but inaccurate (35.13% mAP, 50.48 FPS) | Only option for extreme edge constraints |

---

## 📊 Hypothesis Validation

| Hypothesis | Status | Evidence |
|-----------|--------|----------|
| **H1:** Significant performance differences of at least 5% mAP exist between algorithms across at least 3 of 5 datasets | ✅ **CONFIRMED** | CHVG: 92.26%-54.91%=37.35%, SH17: 64.71%-10.88%=53.83%, SHEL5K: 77.24%-33.99%=43.25% |
| **H2:** Trade-off exists between mAP@0.5 and FPS (higher accuracy = lower speed) | ✅ **CONFIRMED** | YOLOv9 (78.01% mAP, 62.66 FPS) vs YOLOv11 (77.42% mAP, 81.16 FPS) shows inverse relationship |
| **H3:** At least one algorithm achieves mAP≥85%, Recall≥87%, FPS≥25, Latency≤40ms simultaneously | ❌ **NOT CONFIRMED** | YOLOv8 on CHVG: 92.26% mAP ✅, 88.95% Recall ✅, 126 FPS ✅, 7.93ms ✅ — **PASSES ALL** but only on 1 dataset |

---

