# CLIP_COVID

**CLIP_COVID** is a research-oriented 
project that explores the use of 
**CLIP (Contrastive Language–Image Pretraining)** 
for COVID-19–related medical image understanding,
with a focus on **zero-shot inference** and **vision–language alignment**.

The repository contains scripts for zero-shot classification,
caption generation (for fine-tuning CLIP), dataset handling, and evaluation, primarily targeting medical images such as chest X-rays.


---



## Background

Large vision–language models such as **CLIP** enable
zero-shot inference by learning aligned image and text embeddings
from large-scale pretraining.  
This project investigates
how well such models generalize to
**medical imaging domains**, particularly COVID-19 chest X-ray data,
where distribution shift and domain mismatch are common challenges.


## Results and explanation


| Method                                        | Acc   | f1-score | Precision | Recall |
|-----------------------------------------------|-------|----------|-----------|--------|
| CLIP (openAI)                                 | 31.57 | 16.35    | 28.23     |33.36|
| CLIP ( fine-tuned)                            | 57.72 | 54.47    | 59.62     |58.01|
| CLIPVisionEncoder (trained a classifier head) | 87.8  | 87.78    | 88.25     |88.0|


The table above summarizes the performance of different CLIP-based methods
on COVID-19 chest X-ray classification tasks.
The results indicate that fine-tuning the CLIP model significantly improves classification accuracy compared to using the pre-trained model directly.
Additionally, training a classifier head on top of the CLIP vision encoder yields the best performance, 
demonstrating the effectiveness of adapting pre-trained models to specific medical imaging tasks.
## Installation

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

##  Usage

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

## Dataset Assumptions

- Images stored in a directory (e.g., `.png`, `.jpg`)
- Optional annotation files in CSV or JSON format
- Dataset loading logic is defined in `dataset_repo.py`

You can adapt this file to support public datasets such as:
- COVIDx
- BIMCV COVID-19
- NIH ChestXray14 (for transfer experiments)

---


---

##  Limitations
- This project is **for research purposes only** and not for clinical use

---
