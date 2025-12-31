# CLIP_COVID

**CLIP_COVID** is a research-oriented 
project that explores the use of 
**CLIP (Contrastive Language–Image Pretraining)** 
for COVID-19–related medical image understanding,
with a focus on **zero-shot inference** and **vision–language alignment**.

The repository contains scripts for zero-shot classification,
caption generation, dataset handling, and evaluation, primarily targeting medical images such as chest X-rays.

---



## 🧠 Background

Large vision–language models such as **CLIP** enable
zero-shot inference by learning aligned image and text embeddings
from large-scale pretraining.  
This project investigates
how well such models generalize to
**medical imaging domains**, particularly COVID-19 chest X-ray data,
where distribution shift and domain mismatch are common challenges.


## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/mahdaneh/CLIP_COVID.git
cd CLIP_COVID
```

(Optional) Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## 🚀 Usage

### 🔍 Zero-Shot Image Classification

Run CLIP-based classification without fine-tuning of vision or text encoders:

```bash
python classifier_CLIP_visionmodel.py   

---


### 📊 Zero-Shot Evaluation

Evaluate predictions against ground truth:

```bash
python evaluation_zeroshot.py   --predictions preds.json   --ground_truth gt.json
```

Metrics may include accuracy, similarity scores, or ranking-based measures depending on configuration.

---

## 📂 Dataset Assumptions

- Images stored in a directory (e.g., `.png`, `.jpg`)
- Optional annotation files in CSV or JSON format
- Dataset loading logic is defined in `dataset_repo.py`

You can adapt this file to support public datasets such as:
- COVIDx
- BIMCV COVID-19
- NIH ChestXray14 (for transfer experiments)

---

## 📈 Example Output

Example zero-shot prediction:

```json
{
  "image": "patient_012.png",
  "prediction": "COVID-19",
  "confidence": 0.62
}
```

Example caption:

> “Chest X-ray showing bilateral lung opacities suggestive of viral pneumonia.”

---

## ⚠️ Limitations
- This project is **for research purposes only** and not for clinical use

---
