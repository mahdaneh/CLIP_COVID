# CLIP for detecting COVID

This is a research-oriented 
project that explores the use of 
**CLIP (Contrastive Language–Image Pretraining)** 
for COVID-19–related medical image understanding,
with a focus on **zero-shot inference** and **vision–language alignment**.
We used COVID-QU-Ex dataset (contains training, validation, and test sets), and the task is to classify x-ray images into **covid**, **normal**, and **non-covid(Pneumonia)**.
To read more about the used dataset, read "Dataset" section [here](https://github.com/mahdaneh/COVID_transformer).

The repository contains scripts for fine-tuning of a CLIP model (using LORA), zero-shot classification,
caption generation by BLIP (for fine-tuning CLIP), dataset handling, and evaluation, primarily targeting medical images such as chest X-rays. 


## Background

Large vision–language models such as **CLIP** enable
zero-shot inference by learning aligned image and text embeddings
from large-scale pretraining.  
This project investigates
how well such models generalize to
**medical imaging domains**, particularly COVID-19 chest X-ray data,
where distribution shift and domain mismatch are common challenges.

## Methodology
To evaluate zero-shot ability of a pre-trained CLIP (weights from openAI), we use it as is on the test set to classify chest x-ray images (first row in table).

Then, we fine-tune the entire CLIP model (both vision and text encoders) on the training set of COVID x-ray images with corresponding text labels (used BLIP for caption generation, we simply add the true class of each image to its generated caption).
For fine-tuning, we use LORA with rank 4(r=4) for both vision and text encoders, more precisely the weights of self-attention layers, i.e. `v_proj, q_proj` and MLP layers, i.e. `fc1, fc2` are updated during training.
To accommodate data on NVIDIA RTX 4060 Laptop GPU, we have to use a small batch size (80) during fine-tuning of CLIP. 
Note, to have the effect of a larger batch size, we could not use batch accumulation as contrastive loss is batch dependent. To see the result, look at the 2nd row of the following table.

Finally, we extract image embeddings from the CLIP vision encoder (weights are frozen) and train a simple classifier head (a linear layer followed by softmax) 
on top of CLS token by the encoder to classify the images.


As the results in the following table indicate, the classifier head trained on top of the CLIP vision encoder outperforms both zero-shot and fine-tuned CLIP models. Note due to CPU's memory restriction, we could not fine-tune CLIP with larger batch sizes, which might have improved its performance.

## Results and explanation


| Method                                                 | Acc   | f1-score | Precision | Recall |
|--------------------------------------------------------|-------|----------|-----------|--------|
| CLIP (openAI)                                          | 31.57 | 16.35    | 28.23     |33.36|
| CLIP ( fine-tuned)                                     | 57.72 | 54.47    | 59.62     |58.01|
| Classifier-- CLIPVisionEncoder fixed | 87.8  | 87.78    | 88.25     |88.0|


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
