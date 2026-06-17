FASTER R-CNN — HARDWARE LIMITATION NOTICE
==========================================
These folders contain incomplete training attempts for Faster R-CNN.

Training could not be completed due to GPU memory limitations
(NVIDIA RTX 4050 Laptop GPU, 6GB VRAM).

The two-stage region-proposal architecture of Faster R-CNN requires
substantially more VRAM per training sample than the available hardware
could support, even after reducing batch size and image resolution.

This hardware limitation is formally documented in the dissertation:
  → Chapter 4, Section 4.4.3 (Hardware Limitations)

No valid evaluation results were obtained for Faster R-CNN.
It is excluded from all comparative analysis in Chapter 4.

Folders present:
  exp16_fasterrcnn_roboflow/  — incomplete attempt (0% mAP)
  exp17_fasterrcnn_chvg/      — incomplete attempt (0% mAP)
