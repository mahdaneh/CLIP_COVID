from transformers import CLIPModel
import torch
import torch.nn as nn
from tqdm import tqdm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import dataset_repo as d_repo
import json
import albumentations as A
import trackio as wandb
import torch.nn.functional as F
import numpy as np


class CLIPImageClassifier(nn.Module):
    def __init__(self, clip_model, num_classes):
        super().__init__()

        self.image_encoder = clip_model.vision_model
        self.classifier = nn.Linear(
            clip_model.config.vision_config.hidden_size, num_classes, bias=True
        )

    def forward(self, pixel_values):
        outputs = self.image_encoder(pixel_values=pixel_values)
        # CLIP ViT returns 'last_hidden_state'; use [CLS] token embedding
        pooled_output = outputs.pooler_output  # CLS token that post layer norm applied https://github.com/huggingface/transformers/blob/95754b47a6d4fbdad3440a45762531e8c471c528/src/transformers/models/clip/modeling_clip.py#L865
        logits = self.classifier(pooled_output)
        return logits


def main():
    config_file = "config_1.json"
    with open(config_file, "r") as cf:
        config = json.load(cf)

    info = config["INFO"]
    lr = info["LR"]
    total_epoch = info["epochs"]
    bs = info["batch size"]

    wandb.init(
        project="CLIP_COVID",
        name=f"classifier_CLIP_visionmodel_{lr}_{total_epoch}_{bs}_{np.random.randint(0, total_epoch)}",
    )

    transform = A.Compose(
        [
            A.RandomCropFromBorders(),
            A.HorizontalFlip(p=0.5),
            A.Resize(224, 224),
            A.Normalize(),
            A.ToTensorV2(),
        ]
    )

    train_dataset = d_repo.Covid_QU_Ex(
        config["DATA"], mode="train", transform=transform
    )

    tr_loader = DataLoader(
        train_dataset,
        batch_size=bs,
        shuffle=True,
        num_workers=16,
        pin_memory=True,
        persistent_workers=True,
    )

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

    num_classes = 3
    model_ft = CLIPImageClassifier(model, num_classes)

    for param in model_ft.image_encoder.parameters():
        param.requires_grad = False

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_ft.to(device)
    model_ft.train()
    torch.save(model_ft.state_dict(), f"clip_visionmodel_classifier_{0}.pth")
    optimizer = torch.optim.AdamW(model_ft.parameters(), lr=lr)

    CELoss = nn.CrossEntropyLoss()

    for epoch in range(total_epoch):  # number of epochs
        total_loss = 0
        total_correct = 0
        pbar = tqdm(tr_loader)
        for b, data in enumerate(pbar):
            images = data["image"]
            labels = data["class"]

            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model_ft(images)
            loss = CELoss(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            probabilities = F.softmax(logits, dim=1)
            class_prediction = torch.argmax(probabilities, dim=1)

            total_correct += torch.mean(
                torch.where(class_prediction == labels, 1.0, 0.0)
            ).item()
            pbar.set_description(
                f"Loss: {loss.item():.4f}, Acc: {total_correct / (b + 1):.4f}"
            )

        wandb.log(
            {
                "epoch": epoch + 1,
                "loss": total_loss / len(tr_loader),
                "accuracy": total_correct / len(tr_loader),
            }
        )
        print(
            f"Epoch {epoch + 1}, Loss: {total_loss / len(tr_loader):.4f}, Accuracy: {total_correct / len(tr_loader):.4f}"
        )
    wandb.finish()
    torch.save(model_ft.state_dict(), f"clip_visionmodel_classifier_{total_epoch}.pth")


if __name__ == "__main__":
    main()
