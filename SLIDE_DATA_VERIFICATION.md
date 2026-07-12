# FYP Presentation Slides — Data Accuracy Verification
**Verification Date:** 2026-06-29  
**Verified Against:** ALL_EXPERIMENTS_SUMMARY.txt, README.md, Individual Metrics Files

---

## 🔍 VERIFICATION RESULTS

### ✅ SLIDE 11: Complete Results — All 20 Experiments (CORRECT)

**Results Table on Slide:**
| Algorithm | Avg mAP@0.5 | Avg Precision | Avg Recall | Avg F1 | Avg FPS | Avg Latency |
|-----------|-------------|---------------|------------|--------|---------|-------------|
| YOLOv8    | 78.63%      | 82.34%        | 74.93%     | 0.7794 | 78.7    | 35.38       |
| YOLOv9    | 78.01%      | 83.80%        | 73.85%     | 0.7837 | 62.7    | 41.21       |
| YOLOv11   | 77.42%      | 83.01%        | 73.43%     | 0.7754 | 81.2    | 35.96       |
| SSD MobileNet V2 | 35.13% | 69.69%      | 42.91%     | 0.5204 | 50.5    | 19.92       |

**Actual Data from ALL_EXPERIMENTS_SUMMARY.txt:**
| Algorithm | Avg mAP@0.5 | Avg Precision | Avg Recall | Avg F1 | Avg FPS | Avg Latency |
|-----------|-------------|---------------|------------|--------|---------|-------------|
| YOLOv8    | 78.63%      | 82.34%        | 74.93%     | 0.7794 | 78.74   | 35.38       |
| YOLOv9    | 78.01%      | 83.80%        | 73.85%     | 0.7837 | 62.66   | 41.21       |
| YOLOv11   | 77.42%      | 83.01%        | 73.43%     | 0.7754 | 81.16   | 35.96       |
| SSD300    | 35.13%      | 69.69%        | 42.91%     | 0.5204 | 50.48   | 19.92       |

**Status:** ✅ MATCH (FPS values are rounded; all data correct)

---

### ✅ SLIDE 8: Datasets Table (CORRECT)

**Slide Data:**
| Dataset   | Domain           | Images | Classes |
|-----------|------------------|--------|---------|
| SH17      | Manufacturing    | 8,099  | 17      |
| SFCHD     | Chemical Plant   | 12,373 | 7       |
| CHVG      | Construction     | 1,699  | 8       |
| SHEL5K    | Helmet-focused   | 5,000  | 6       |
| Roboflow  | Construction     | 1,206  | 5       |

**Status:** ✅ MATCH with experiment configuration files

---

### ✅ SLIDE 12: Comparative Analysis Key Patterns (CORRECT)

**Slide Claims:**
- YOLO inter-generation mAP gap only 1.21% → `78.63% - 77.42% = 1.21%` ✅
- YOLOv9 best Precision (83.80%) ✅
- YOLOv11 highest FPS (81.2) ✅
- SSD MobileNet V2 maintained stable FPS (43–53) ✅

**Status:** ✅ ALL CORRECT

---

### ✅ SLIDE 13: Multi-Criteria Evaluation Weights (CORRECT)

**Weight Distribution on Slide:**
- Recall: 30%
- mAP@0.5: 20%
- F1-Score: 15%
- FPS: 15%
- Precision: 10%
- Latency: 10%

**Normalized Scores Summary on Slide:**
| Algorithm | Total Score |
|-----------|-------------|
| YOLOv8    | 0.9024      |
| YOLOv11   | 0.8947      |
| YOLOv9    | 0.7967      |
| SSD MobileNet V2 | 0.1000 |

**Actual Data from ALL_EXPERIMENTS_SUMMARY.txt FINAL RANKING:**
| Algorithm | Final Score |
|-----------|-------------|
| YOLOv8    | **0.9024**  |
| YOLOv11   | **0.8947**  |
| YOLOv9    | **0.7967**  |
| SSD300    | **0.1000**  |

**Status:** ✅ MATCH — All scores correct

---

### ❌ **CRITICAL ERROR FOUND: SLIDE 14 — Final Algorithm Ranking Pyramid (INCORRECT)**

**Pyramid Visualization on Slide 14 shows:**
| Rank | Algorithm | Score | Details |
|------|-----------|-------|---------|
| 1st  | YOLOv8    | **0.893** | mAP@0.5: 0.893 |
| 2nd  | YOLOv11   | **0.867** | mAP@0.5: 0.867 |
| 3rd  | YOLOv9    | **0.821** | mAP@0.5: 0.821 |
| 4th  | SSD MobileNet V2 | **0.732** | mAP@0.5: 0.732 |

**Actual Final Ranking Scores (from ALL_EXPERIMENTS_SUMMARY.txt):**
| Rank | Algorithm | Score | mAP@0.5 |
|------|-----------|-------|---------|
| 1st  | YOLOv8    | **0.9024** | 78.63% |
| 2nd  | YOLOv11   | **0.8947** | 77.42% |
| 3rd  | YOLOv9    | **0.7967** | 78.01% |
| 4th  | SSD300    | **0.1000** | 35.13% |

**Issues Identified:**
1. **Wrong scoring in pyramid:**
   - Shows 0.893, 0.867, 0.821, 0.732
   - Should be: 0.9024, 0.8947, 0.7967, 0.1000
   - The pyramid scores appear to be raw metric values, NOT weighted final scores

2. **Also shows wrong mAP values in the pyramid details:**
   - Shows 0.893, 0.867, 0.821, 0.732 as mAP values
   - Should be: 78.63%, 77.42%, 78.01%, 35.13%

**Status:** ❌ **INCORRECT — Slide 14 pyramid has wrong data**

---

### ✅ SLIDE 10: Results Summary Table (CORRECT)

**Slide Table matches the main results section exactly** ✅

---

### ✅ SLIDE 1 & OVERVIEW INFORMATION (CORRECT)

- Presentation title: ✅
- Student: Chan Heng Yi ✅
- Supervisor: Dr. Subarmaniam A/L Kannan ✅
- Institution: Multimedia University (MMU), FIST ✅
- Hardware: NVIDIA RTX 4050, AMD Ryzen 7 7445HS, 16GB RAM ✅

---

## 📊 SUMMARY

| Slide | Content | Status |
|-------|---------|--------|
| 8 | Datasets Table | ✅ CORRECT |
| 10 | Results Summary | ✅ CORRECT |
| 11 | Complete Results (All 20 Exp) | ✅ CORRECT |
| 12 | Comparative Analysis | ✅ CORRECT |
| 13 | Multi-Criteria Evaluation & Scores | ✅ CORRECT |
| **14** | **Final Ranking Pyramid** | ❌ **INCORRECT** |

---

## 🔧 CORRECTION NEEDED

**Slide 14 — Final Algorithm Ranking Pyramid visualization:**

The pyramid currently shows incorrect scores (0.893, 0.867, 0.821, 0.732) and mAP values.

**Should be updated to show:**
```
1st  YOLOv8    — Score: 0.9024, mAP@0.5: 78.63%, Recall: 74.93%
2nd  YOLOv11   — Score: 0.8947, mAP@0.5: 77.42%, FPS: 81.16 (Highest)
3rd  YOLOv9    — Score: 0.7967, mAP@0.5: 78.01%, Precision: 83.80%
4th  SSD MobileNet V2 — Score: 0.1000, mAP@0.5: 35.13%, Latency: 19.92ms
```

---

**Overall Accuracy: 6 out of 7 major slide sections are correct (85.7%)**  
**Critical Issue: Slide 14 requires correction before final presentation**
