# CLIP for COVID-19 Detection from Chest X-rays

This experimental project explores the use of **CLIP (Contrastive Language–Image Pretraining)** for COVID-19 medical image understanding, focusing on **zero-shot inference** and **vision–language alignment**.  

We show that **zero-shot CLIP alone is insufficient** for COVID-19 chest X-ray classification due to domain shift: medical images differ significantly from the natural images CLIP was trained on.

---

## Dataset

We used the **COVID-QU-Ex dataset**, which includes training, validation, and test sets. The task is to classify X-ray images into:

- `COVID`  
- `Normal`  
- `Non-COVID (Pneumonia)`

For details on dataset preprocessing and splits, see the [COVID Transformer repository](https://github.com/mahdaneh/COVID_transformer).  

The repository includes scripts for:

- Fine-tuning CLIP (with LoRA)  
- Zero-shot classification  
- Caption generation using BLIP (for CLIP fine-tuning)  
- Dataset handling and evaluation

---

## Background

Large vision–language models like **CLIP** enable zero-shot inference by learning aligned image and text embeddings from large-scale pretraining.  

This project investigates **how well CLIP generalizes to medical imaging**, particularly COVID-19 chest X-rays, where **domain shift** is a major challenge.

---

## Methodology

1. **Zero-shot CLIP:**  
   Applied pre-trained CLIP directly on the test set to classify chest X-ray images.

2. **Fine-tuning CLIP:**  
   - Fine-tuned the full CLIP model (vision + text encoders) on the training set.  
   - Used BLIP for caption generation and appended the true class label to each image caption.  
   - Applied **LoRA** (rank `r=4`) to self-attention layers (`v_proj`, `q_proj`) and MLP layers (`fc1`, `fc2`).  
   - Batch size was limited to 80 due to GPU memory (NVIDIA RTX 4060 Laptop). Batch accumulation could not be used due to contrastive loss dependence on batch statistics.

3. **Classifier on CLIP Vision Encoder:**  
   - Used the CLS token of the CLIP vision encoder (frozen)  
   - Trained a simple **linear classifier with softmax** on top  
   - This approach outperformed both zero-shot and fine-tuned CLIP models

---

## Results

| Method                             | Accuracy | F1-score | Precision | Recall |
|------------------------------------|---------|----------|-----------|--------|
| CLIP (zero-shot)                    | 31.57   | 16.35    | 28.23     | 33.36 |
| CLIP (fine-tuned)                   | 57.72   | 54.47    | 59.62     | 58.01 |
| Classifier on CLIP Vision Encoder   | 87.8    | 87.78    | 88.25     | 88.0  |

---

## Conclusion

- **Zero-shot CLIP is insufficient** for COVID-19 X-ray classification due to domain shift.  
- Fine-tuning CLIP improves results, but the **best performance comes from a classifier on top of the frozen CLIP vision encoder**.  
- "Classifier on CLIP Vision Encoder " is especially practical for **limited-resource devices** with low GPU memory.  
- Overall, CLIP provides a strong foundation, but **specialized adaptation is necessary** for medical imaging tasks.

---

**Key Insight:**  
Zero-shot inference **cannot provide consistent accuracy** in domains where the data distribution differs from pretraining (e.g., medical images vs. natural images).