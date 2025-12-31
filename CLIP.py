from PIL import Image
import requests
import os
import torch
import torch.nn.functional as F

from transformers import CLIPProcessor, CLIPModel

from peft import PeftModel, LoraConfig, get_peft_model
from torch.utils.data import DataLoader
import json
from torch.utils.data import Dataset

import pandas as pd
import numpy as np
import albumentations as A
from tqdm import tqdm
import trackio as wandb


class DictTorchData(Dataset):
    def __init__(self, data_list, transform=None) -> None:
        super().__init__()
        self.data_list = data_list
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        image = Image.open(item["image"]).convert("RGB")
        image = np.array(image)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        text = item["text"]
        return {"image": image, "text": text}


def train():
    config_file = "config_3.json"
    with open(config_file, "r") as f:
        config = json.load(f)
    name = f"CLIP_fineTuneLora{config['INFO']['batch size']}_{config['INFO']['epochs']}_{config['LORA']['target_modules']}"
    wandb.init(project="CLIP_LoRA_COVID", name=name)
    processor = CLIPProcessor.from_pretrained("clip-vit-base-patch32", use_fast=True)

    # load dataset
    tr_json = "train_HF_transformers.jsonl"
    val_json = "val_HF_transformers.jsonl"

    df = pd.read_json(tr_json, lines=True)
    training_list = df.to_dict(orient="records")
    tr_dataset = DictTorchData(training_list)

    df = pd.read_json(val_json, lines=True)
    val_list = df.to_dict(orient="records")
    val_dataset = DictTorchData(val_list)

    def collate_fn(batch):
        images = [x["image"] for x in batch]
        texts = [x["text"] for x in batch]

        return processor(images=images, text=texts, return_tensors="pt", padding=True)

    tr_loader = DataLoader(
        tr_dataset,
        batch_size=config["INFO"]["batch size"],
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn
    )

    # outputs = model(**inputs)
    # logits_per_image = outputs.logits_per_image # this is the image-text similarity score
    # probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities

    lora_config = LoraConfig(
        r=4,  # rank
        lora_alpha=16,
        use_rslora=True,
        target_modules=config["LORA"]["target_modules"],
        lora_dropout=0.1,
        bias="none",
        task_type=None,
    )
    # load CLIP model and processor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

    model = get_peft_model(model, lora_config)
    model.logit_scale.requires_grad = False

    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    model.print_trainable_parameters()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,  # higher LR is OK for LoRA
        weight_decay=1e-4,
    )

    model.to(device)
    model.train()

    for epoch in range(config["INFO"]["epochs"]):
        total_loss = 0
        total_recall = 0
        pbar = tqdm(tr_loader)
        for i, batch in enumerate(pbar):
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch, return_loss=True)
            # import pdb; pdb.set_trace()
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            total_recall += recall_at_1(outputs)
            pbar.set_description(
                f"Epoch {epoch} batch {i + 1}: "
                f"loss = {total_loss / (i + 1):.4f}"
                f", recall@1 = {total_recall / (i + 1):.4f}"
            )
        total_loss /= i + 1
        total_recall /= i + 1

        # val_recall = eval(model, val_loader, device)
        wandb.log(
            {
                "epoch": epoch + 1,
                "train_loss": total_loss,
                "train_recall@1": total_recall,
                # "val_recall@1": val_recall
            }
        )
        if (epoch + 1) % 10 == 0:
            model.save_pretrained("clip_lora_weights/name")
    wandb.finish()


def eval(model, val_loader, device):
    model.eval()
    pbar = tqdm(val_loader)
    total = 0
    for i, batch in enumerate(pbar):
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(**batch, return_loss=True)
        total += recall_at_1(outputs)

        pbar.set_description(f"Eval batch {i + 1}: recall = {total / (i + 1):.4f}")
    return total / (i + 1)


def compute_loss(outputs, model):
    image_embeds = F.normalize(outputs.image_embeds, dim=-1)
    text_embeds = F.normalize(outputs.text_embeds, dim=-1)

    logit_scale = model.logit_scale.exp()
    logits = logit_scale * image_embeds @ text_embeds.T

    labels = torch.arange(logits.size(0), device=logits.device)

    loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2

    loss.backward()


def recall_at_1(outputs):
    # Normalize embeddings
    image_features = F.normalize(outputs.image_embeds, dim=1)
    text_features = F.normalize(outputs.text_embeds, dim=1)

    # Compute cosine similarity [N_images, N_texts]
    sims = image_features @ text_features.T

    # For each image, rank texts by similarity
    top1 = sims.argmax(dim=1)  # index of most similar text for each image

    # Correct if the top-1 index matches the ground truth (assume diagonal = correct)
    correct = torch.arange(len(image_features), device=image_features.device)
    recall1 = (top1 == correct).float().mean().item()

    return recall1


if __name__ == "__main__":
    train()
