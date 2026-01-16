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
   - Fine-tuned the full CLIP model (vision + text encoders) on the training set using LORA (r=4) on (`v_proj`, `q_proj`) and MLP layers (`fc1`, `fc2`)  
   - Used BLIP for caption generation and appended the true class label to each image caption.
   - Batch size was limited to 80 due to GPU memory (NVIDIA RTX 4060 Laptop). Batch accumulation could not be used due to contrastive loss dependence on batch statistics.

3. **Classifier on CLIP Vision Encoder:**  
   - Used the CLS token of the CLIP vision encoder (frozen). We used CLIP models pretrained on both ImageNet (CLIP_ImageNet)
   and fine-tuned CLIP (i.e. method 2) on X-ray datasets(CLIP_Xray).
   - Trained a simple **linear classifier with softmax** on top  
   - This approach outperformed both zero-shot and fine-tuned CLIP models

---

## Results
Besides comparing the methods mentioned above, we also considered
baseline CNN model (Resnet18) and ViT_b_16, which are fine-tuned on the same dataset (
Read more [here](https://github.com/mahdaneh/COVID_transformer) ).

Clearly, the pretrained CLIP model struggles with
zero-shot classification- CLIP (zero-shot)- due to domain shift.
Although fine-tuning CLIP 
with LORA (batchsize=80) substantially enhances its performance,
still notably below the fully supervised end-to-end models (ViT_b_16 and Resnet18). 
This might be caused by the limited batch size during fine-tuning, which is crucial for contrastive learning. We could not use larger batch sizes due 
to GPU memory constraints.

We found training a simple classifier on top of the frozen fully pretrained CLIP (CLIP_ImageNet) vision encoder
yields the promising results, explaining that CLIP's visual representations are **strong and transferable**.
However, the best performance is achieved by the classifier on top of fined-tuned CLIP (CLIP_Xray) vision encoder,
indicating that **adapting the vision encoder** to the medical domain further boosts performance. 
This shows partially training (with LORA) CLIP text and vision encoders on the target domain (Xray images)
can lead to a better aligned embedded
feature space, resulting in improved classification performance.

These findings suggest:
- **CLIP’s visual representations are strong and transferable**

- The main limitation lies in zero-shot text–image matching, not the visual features themselves

- Decoupling the vision encoder from the text encoder allows better task-specific discrimination
- Domain adaptation of the vision encoder yields substantial performance gains

**Main conclusion**: This result highlights that CLIP
is most effective as a feature extractor, rather than as a zero-shot classifier for this task.


| Method                                   | Accuracy  | F1-score  | Precision | Recall    |
|------------------------------------------|-----------|-----------|-----------|-----------|
| Resnet18 (fine-tuned)                    | 82.56     | 82.37     | 82.66     | 82.41     |
| ViT_b_16 (fine-tuned)                    | 87.51     | 82.37     | 87.45     | 87.37     |
| CLIP_ImageNet (zero-shot)                | 31.57     | 16.35     | 28.23     | 33.36     |
| CLIP_Xray (fine-tuned,LORA,bs=80)        | 57.72     | 54.47     | 59.62     | 58.01     |
| Classifier on CLIP_ImageNet Vision Encoder | 87.8      | 87.78     | 88.25     | 88.0      |
| Classifier on CLIP_Xray Vision Encoder   | **92.44** | **92.41** | **92.55** | **92.51** |
---

## Conclusion

- **Zero-shot CLIP is insufficient** for COVID-19 X-ray classification due to domain shift.  
- Fine-tuning CLIP improves results, but the **best performance comes from a classifier on top of the frozen CLIP vision encoder**.  
- "Classifier on CLIP Vision Encoder " is especially practical for **limited-resource devices** with low GPU memory.  
- Overall, CLIP provides a strong foundation, but **specialized adaptation is necessary** for medical imaging tasks.

---

**Key Insight:**  
Zero-shot inference **cannot provide consistent accuracy** in domains where the data distribution differs from pretraining (e.g., medical images vs. natural images).