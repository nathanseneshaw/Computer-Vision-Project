# Computer Vision Project

Road damage detection with [Ultralytics YOLO](https://docs.ultralytics.com/) and the
[RDD2022 dataset](https://doi.org/10.6084/m9.figshare.21431547).

## Setup

**1. Install YOLO**

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install ultralytics
```

On macOS/Linux, activate with `source .venv/bin/activate` instead.

**2. Download the dataset**

```powershell
python scripts/download_rdd2022.py
```

This downloads RDD2022 (13 GB) into `data/RDD2022/`. If the download is interrupted, run the command again to resume.
