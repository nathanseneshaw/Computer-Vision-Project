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
python download_rdd2022.py
```

This downloads RDD2022 (13 GB) into `data/RDD2022/`. If the download is interrupted, run the command again to resume.


**3. Running the Interface**

Download Gradio 
```powershell
pip install ultralytics gradio
```
Train the model 
```powershell
python src/prepare_labels.py
python src/train_ultralytics.py
```

Run the interface
```powershell
python src/app.py
```

Open the link it prints (http://127.0.0.1:7860), upload a road photo and click **Analyze**. Damage is boxed on the image, coloured by severity (yellow minor, orange moderate, red severe), and the image gets an overall rating.

It loads the newest `best.pt` under `models/ultralytics/`. To use a different model: `python src/app.py --weights path/to/best.pt`.

**How severity works.** RDD2022 doesn't label severity, so it is a heuristic in `src/severity.py`: a 0-100 score from how much of the image the damage covers, the size of the worst single defect, and the number of defects, weighted by damage type. Treat it as a rough triage aid rather than an engineering measure. The thresholds are constants at the top of that file.

**Damage types.** The model only finds what it was trained on. `train_ultralytics.py` currently trains potholes only (D40). The app and severity scoring read class names from the model, so a model trained on cracks too (D00, D10, D20) will work without changes.

