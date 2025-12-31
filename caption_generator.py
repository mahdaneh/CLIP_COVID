from tqdm import tqdm
import dataset_repo as d_repo
from torch.utils.data import DataLoader
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import json
import glob
import os

device = "cuda"


def main():
    # Load BLIP model

    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    ).to(device)
    model.eval()

    data_path = "/home/mahdieh/Documents/Datasets/COVID/COVID-QU-Ex/Lung_Segmentation_Data/Train/COVID/images"

    config_file = "config_1.json"

    with open(config_file, "r") as cf:
        config = json.load(cf)

    dataset = d_repo.Covid_QU_Ex(
        config["DATA"],
        mode="test",
    )
    val_dataset = d_repo.Covid_QU_Ex(
        config["DATA"],
        mode="val",
    )

    loader = DataLoader(
        dataset,
        batch_size=config["INFO"]["batch size"],
        shuffle=True,
        num_workers=16,
        pin_memory=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["INFO"]["batch size"],
        num_workers=16,
        pin_memory=True,
        persistent_workers=True,
    )
    write_image_text_json("test_HF_transformers", loader, processor, model)
    write_image_text_json("val_HF_transformers", val_loader, processor, model)


from dataset_repo import *


def write_image_text_json(json_fname, data_loader, processor, model):
    # Output file
    jsonl_path = f"{json_fname}.jsonl"

    # Remove old file if exists
    if os.path.exists(jsonl_path):
        os.remove(jsonl_path)

    with open(jsonl_path, "w") as f:
        for b, data in tqdm(enumerate(data_loader)):
            images = data["image"]
            inputs_paths = data["image_path"]
            label_name = data["label"]
            inputs = processor(images, return_tensors="pt")
            inputs.to(device)

            outputs = model.generate(
                **inputs, max_new_tokens=50, num_beams=1, do_sample=False
            )
            captions = processor.batch_decode(outputs, skip_special_tokens=True)

            for img_path, cls_name, text in zip(inputs_paths, label_name, captions):
                entry = {
                    "image": img_path,
                    "text": text + " " + cls_name,
                }
                f.write(json.dumps(entry) + "\n")


if __name__ == "__main__":
    main()
