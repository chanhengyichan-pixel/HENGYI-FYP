# FYP Thesis Section 4.5 - Experimental Results Checklist

## Overview
This checklist documents everything needed for thesis section 4.5 (Experimental Results).

**Total Algorithms:** 4 (YOLOv8, YOLOv9, YOLOv11, SSD300)  
**Total Datasets:** 5 (Roboflow, CHVG, SHEL5K, SH17, SFCHD)  
**Total Experiment Combinations:** 20 (4 algorithms × 5 datasets)

---

## 4.5.1 YOLOv8 Results (EXP 01-05)

### Required Metrics (per dataset):
- [ ] mAP@0.5 (%)
- [ ] mAP@0.5:0.95 (%)
- [ ] Precision (%)
- [ ] Recall (%)
- [ ] F1-Score
- [ ] FPS (frames per second)
- [ ] Latency (ms)
- [ ] Training time (hours)
- [ ] Per-class AP@0.5 (all classes in dataset)
- [ ] Confusion matrix or class breakdown

### Files to Collect:
- [ ] EXP 01 metrics (Roboflow) — `/home/heng/FYP/results/YOLOv8/exp01_roboflow/metrics.txt`
- [ ] EXP 02 metrics (CHVG)
- [ ] EXP 03 metrics (SHEL5K)
- [ ] EXP 04 metrics (SH17)
- [ ] EXP 05 metrics (SFCHD)

### Tables to Create:
- [ ] Table 4.1: YOLOv8 Performance Summary (5 rows × 7 columns)
  ```
  | Dataset | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | F1 | FPS | Latency |
  |---------|---------|--------------|-----------|--------|----|----- -------|
  ```
- [ ] Table 4.2: YOLOv8 Per-Class AP (5 rows for Roboflow classes example)

### Figures to Create:
- [ ] Figure 4.1: YOLOv8 mAP@0.5 across datasets (bar chart)
- [ ] Figure 4.2: YOLOv8 Precision vs Recall across datasets (line chart)
- [ ] Figure 4.3: YOLOv8 Per-class AP breakdown (grouped bar chart, per dataset or selected)

---

## 4.5.2 YOLOv9 Results (EXP 06-10)

### Required Metrics (same as above):
- [ ] mAP@0.5, mAP@0.5:0.95, Precision, Recall, F1, FPS, Latency per dataset
- [ ] Per-class AP@0.5 for each dataset

### Files to Collect:
- [ ] EXP 06 metrics (Roboflow)
- [ ] EXP 07 metrics (CHVG)
- [ ] EXP 08 metrics (SHEL5K)
- [ ] EXP 09 metrics (SH17)
- [ ] EXP 10 metrics (SFCHD)

### Tables to Create:
- [ ] Table 4.3: YOLOv9 Performance Summary (5 rows × 7 columns)
- [ ] Table 4.4: YOLOv9 Per-Class AP (as needed)

### Figures to Create:
- [ ] Figure 4.4: YOLOv9 mAP@0.5 across datasets
- [ ] Figure 4.5: YOLOv9 Precision vs Recall
- [ ] Figure 4.6: YOLOv9 Per-class AP breakdown

---

## 4.5.3 YOLOv11 Results (EXP 11-15)

### Required Metrics (same as above):
- [ ] mAP@0.5, mAP@0.5:0.95, Precision, Recall, F1, FPS, Latency per dataset
- [ ] Per-class AP@0.5 for each dataset

### Files to Collect:
- [ ] EXP 11 metrics (Roboflow)
- [ ] EXP 12 metrics (CHVG)
- [ ] EXP 13 metrics (SHEL5K)
- [ ] EXP 14 metrics (SH17)
- [ ] EXP 15 metrics (SFCHD)

### Tables to Create:
- [ ] Table 4.5: YOLOv11 Performance Summary (5 rows × 7 columns)
- [ ] Table 4.6: YOLOv11 Per-Class AP (as needed)

### Figures to Create:
- [ ] Figure 4.7: YOLOv11 mAP@0.5 across datasets
- [ ] Figure 4.8: YOLOv11 Precision vs Recall
- [ ] Figure 4.9: YOLOv11 Per-class AP breakdown

---

## 4.5.4 SSD300 MobileNetV2 Results (EXP 21-25)

### Required Metrics:
- [ ] mAP@0.5, mAP@0.5:0.95, Precision, Recall, F1, FPS, Latency per dataset
- [ ] Per-class AP@0.5 for each dataset
- [ ] Training epochs, batch size, image size (for context)

### Files to Collect:
- [ ] EXP 21 metrics (Roboflow) — `/home/heng/FYP/results/SSD/exp21_ssd_roboflow/metrics.txt` ✓
- [ ] EXP 22 metrics (CHVG) ✓
- [ ] EXP 23 metrics (SHEL5K) ✓
- [ ] EXP 24 metrics (SH17) ✓ (note: 5 epochs only)
- [ ] EXP 25 metrics (SFCHD) ✓ (note: 5 epochs only)

### Tables to Create:
- [ ] Table 4.7: SSD300 Performance Summary (5 rows × 8 columns, include epochs)
- [ ] Table 4.8: SSD300 Per-Class AP (as needed)
- [ ] Table 4.9: SSD300 Training Configuration (epochs, batch, imgsz)

### Figures to Create:
- [ ] Figure 4.10: SSD300 mAP@0.5 across datasets
- [ ] Figure 4.11: SSD300 Precision vs Recall
- [ ] Figure 4.12: SSD300 Per-class AP breakdown (focused on EXP 21-23 full training)

### Important Note:
- ⚠ EXP 24-25 (SH17, SFCHD) have only 5 epochs with aggressive downgrades
- These are **NOT comparable** to EXP 21-23 (100 epochs)
- Consider separate discussion or note in table

---

## 4.5.5 Faster R-CNN Results (If Applicable)

### Status: ❌ NOT COMPLETED
- Faster R-CNN training failed due to hardware/software issues
- Not included in results

### Decision Options:
- [ ] **Option A: Omit Faster R-CNN** — Focus on 4 completed algorithms
- [ ] **Option B: Add note** — Explain why Faster R-CNN was excluded
- [ ] **Option C: Use pre-trained baseline** — Compare against COCO pre-trained (if needed)

---

## 4.5.6 Cross-Algorithm Comparison Tables

### Required Summary Tables:
- [ ] **Table 4.10: Algorithm Comparison on Roboflow**
  - Rows: YOLOv8, YOLOv9, YOLOv11, SSD300
  - Columns: mAP@0.5, mAP@0.5:0.95, Precision, Recall, F1, FPS, Latency
  
- [ ] **Table 4.11: Algorithm Comparison on CHVG**
  - Same structure as above

- [ ] **Table 4.12: Algorithm Comparison on SHEL5K**
  - Same structure as above

- [ ] **Table 4.13: Algorithm Comparison on SH17**
  - Same structure as above
  - Note: SSD300 only has minimal training (5 epochs)

- [ ] **Table 4.14: Algorithm Comparison on SFCHD**
  - Same structure as above
  - Note: SSD300 only has minimal training (5 epochs)

### Required Summary Figures:
- [ ] **Figure 4.13: Algorithm Performance Comparison (mAP@0.5)**
  - X-axis: Datasets (Roboflow, CHVG, SHEL5K, SH17, SFCHD)
  - Y-axis: mAP@0.5 (%)
  - Lines: YOLOv8, YOLOv9, YOLOv11, SSD300

- [ ] **Figure 4.14: Precision vs Recall Comparison**
  - X-axis: Recall (%)
  - Y-axis: Precision (%)
  - Points/Lines: All algorithms, all datasets

- [ ] **Figure 4.15: Speed-Accuracy Trade-off**
  - X-axis: FPS (speed)
  - Y-axis: mAP@0.5 (accuracy)
  - Points: All algorithms, all datasets (color-coded by algorithm)

- [ ] **Figure 4.16: Per-Dataset Performance Heatmap**
  - Rows: Algorithms (YOLOv8, YOLOv9, YOLOv11, SSD300)
  - Columns: Datasets (Roboflow, CHVG, SHEL5K, SH17, SFCHD)
  - Values: mAP@0.5 (color intensity)

---

## Data Organization Checklist

### Metrics Files Location:
```
✓ /home/heng/FYP/results/YOLOv8/exp*_*/metrics.txt
✓ /home/heng/FYP/results/YOLOv9/exp*_*/metrics.txt
✓ /home/heng/FYP/results/YOLOv11/exp*_*/metrics.txt
✓ /home/heng/FYP/results/SSD/exp*_ssd_*/metrics.txt
```

### Summary Tables Location:
```
✓ /home/heng/FYP/results/SSD_FINAL_COMPARISON_TABLE.txt
? /home/heng/FYP/results/YOLO_FINAL_COMPARISON_TABLE.txt (need to create)
? /home/heng/FYP/results/COMPREHENSIVE_COMPARISON_TABLE.txt (need to create)
```

### Logs Location:
```
✓ /home/heng/FYP/training_*.log (for reference, not for thesis)
✓ /home/heng/FYP/results/*/metrics.txt (main source)
```

---

## Quick Data Gathering Script

```bash
# Collect all metrics
find /home/heng/FYP/results -name "metrics.txt" -exec cat {} \;

# Create summary comparison
python3 create_comprehensive_results_table.py

# Verify all experiments exist
ls -la /home/heng/FYP/results/*/exp*/metrics.txt
```

---

## Thesis Writing Checklist for Section 4.5

### Text Content:
- [ ] 4.5.1 Introduction paragraph (overview of 20 experiments)
- [ ] 4.5.1 YOLOv8 subsection (2-3 paragraphs)
  - [ ] Describe performance across datasets
  - [ ] Highlight best/worst performing datasets
  - [ ] Per-class analysis for selected dataset
  
- [ ] 4.5.2 YOLOv9 subsection (2-3 paragraphs)
  - [ ] Compare to YOLOv8
  - [ ] Performance improvements/degradation

- [ ] 4.5.3 YOLOv11 subsection (2-3 paragraphs)
  - [ ] Compare to YOLOv8 and YOLOv9
  - [ ] Latest version performance

- [ ] 4.5.4 SSD300 subsection (2-3 paragraphs)
  - [ ] Explain architecture difference (MobileNetV2 lightweight)
  - [ ] Note on EXP 24-25 (minimal training, not comparable)
  - [ ] Performance vs YOLO variants

- [ ] 4.5.5 Cross-Algorithm Discussion (2-3 paragraphs)
  - [ ] Which algorithm performs best overall?
  - [ ] Dataset-specific observations
  - [ ] Speed vs accuracy trade-offs
  - [ ] Per-class performance insights

### Tables & Figures:
- [ ] All 16 tables listed above
- [ ] All 16 figures listed above
- [ ] Proper captions for all (descriptive)
- [ ] References in text to each table/figure

---

## Final Deliverables Summary

| Item | Count | Status |
|------|-------|--------|
| Experiments | 20 | ✓ Complete |
| Algorithms | 4 | ✓ Complete |
| Datasets | 5 | ✓ Complete |
| Metrics files | 20 | ✓ Collected |
| Tables to create | 14 | ⏳ Pending |
| Figures to create | 16 | ⏳ Pending |
| Text subsections | 6 | ⏳ Pending |

---

## Files Currently Available

### ✓ Already Have:
- `/home/heng/FYP/results/SSD_FINAL_COMPARISON_TABLE.txt` — SSD results summary
- `/home/heng/FYP/results/SSD_ALL_RESULTS_DETAILED.txt` — Detailed SSD breakdown
- All metrics.txt files in experiment directories

### ⏳ Need to Create:
- YOLO comprehensive comparison table
- Cross-algorithm comparison tables (4.10-4.14)
- Visualization figures (4.1-4.16)

---

## Next Steps

1. **Gather all metrics** ✓ (done)
2. **Create summary tables** (4.10-4.14) — Python script needed
3. **Generate figures** (4.1-4.16) — Plotting needed
4. **Write text sections** (4.5.1-4.5.5) — Narrative needed
5. **Review and integrate** into thesis document

