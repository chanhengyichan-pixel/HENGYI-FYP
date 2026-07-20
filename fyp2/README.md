# Edge AI Industrial Safety Compliance Monitoring

A Smart Safety Access Control System that uses YOLOv8 to verify workers are wearing PPE before entering a hazardous area, with a live web dashboard. Designed to run on a laptop and deploy to Raspberry Pi 5.

## Setup

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

> **Python version:** Developed and tested on Python 3.12.3. Use the same major Python version when deploying to Raspberry Pi to avoid dependency mismatches.
