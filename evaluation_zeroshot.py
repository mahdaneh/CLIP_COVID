import json
import numpy as np
import mlflow
import torch
from ptflops import get_model_complexity_info
from sklearn.metrics import *
from torch.utils.data import DataLoader

from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
from peft import PeftModel, LoraConfig, get_peft_model
from classifier_CLIP_visionmodel import CLIPImageClassifier
import albumentations as A
import dataset_repo as d_repo


def eval_model(processor, model, data_loader, clip_zeroshot=False):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    y_true = []
    y_pred = []
    all_class_names = ["covid", "non-covid", "noraml"]
    for b, data in tqdm(enumerate(data_loader)):
        images = data["image"]
        inputs_paths = data["image_path"]
        label_name = data["label"]
        classes = data["class"]
        if clip_zeroshot:
            inputs = processor(
                text=["chest xray" + cls_nm for cls_nm in all_class_names],
                images=images,
                return_tensors="pt",
                padding=True,
            ).to(device)
            outputs = model(**inputs)
            logits_per_image = (
                outputs.logits_per_image
            )  # this is the image-text similarity score
            probs = logits_per_image.softmax(
                dim=1
            )  # we can take the softmax to get the label probabilities
        else:
            inputs = images.to(device)
            logits = model(inputs)
            probs = torch.softmax(logits, dim=1)

        y_pred.append(probs.detach().cpu().numpy())
        y_true.append(classes)

    # import pdb;pdb.set_trace()
    y_pred = np.vstack(y_pred)
    y_true = np.hstack(y_true)

    return y_true, y_pred, all_class_names, model


def metrics(y_true, y_pred, class_names, experiment_name="None", run_name="None"):
    mlflow.set_experiment(experiment_name)
    # run_name = "CLIP_classifier_visionmodel_eval"
    with mlflow.start_run(run_name=run_name, nested=True):
        metrics_dict = {}
        y_pred_classes = np.argmax(y_pred, axis=1)
        p, r, f1, s = precision_recall_fscore_support(y_true, y_pred_classes)

        accuracy = accuracy_score(y_true, y_pred_classes)
        for i, cls in enumerate(class_names):
            metrics_dict["precision_class_%s" % cls] = p[i]
            metrics_dict["recall_class_%s" % cls] = r[i]
            metrics_dict["f1_class_%s" % cls] = f1[i]
            metrics_dict["support_class_%s" % cls] = s[i]

        p, r, f1, s = precision_recall_fscore_support(
            y_true, y_pred_classes, average="macro"
        )
        metrics_dict["precision_macro"] = p
        metrics_dict["recall_macro"] = r
        metrics_dict["f1_macro"] = f1
        metrics_dict["accuracy"] = accuracy

        mlflow.log_metrics(metrics_dict)
        print(metrics_dict)
        # mlflow.log_text(
        #     str(flops + "/n" + str(params)), "flops_params_%s.txt" % run_name
        # )

        cm = confusion_matrix(y_true, y_pred_classes)

        path_cm = "mlruns/%s_cm.npy" % run_name

        np.save(path_cm, cm)
        mlflow.log_artifact(path_cm)

        # disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
        # disp.plot()
        # plt.savefig("docs/confusion_matrix_%s.png" % name)


def data_preparation(config):
    transform = A.Compose(
        [
            A.Resize(224, 224),
            A.Normalize(),
            A.ToTensorV2(),
        ]
    )
    dataset = d_repo.Covid_QU_Ex(config["DATA"], mode="test", transform=transform)

    loader = DataLoader(
        dataset,
        batch_size=config["INFO"]["batch size"],
        shuffle=True,
        num_workers=16,
        pin_memory=True,
        persistent_workers=True,
    )
    return loader


if __name__ == "__main__":
    config_file = "config_1.json"

    with open(config_file, "r") as cf:
        config = json.load(cf)

    base_model_path = "openai/clip-vit-base-patch32"
    model_path = "clip_visionmodel_classifier_100.pth"
    mlflow_run_name = model_path.split("/")[-1]
    mlflow_experiment_name = "Final_Comparison_Models"

    processor = CLIPProcessor.from_pretrained(base_model_path)
    # model = CLIPModel.from_pretrained(model_path)
    # clip_zeroshot = True

    base_model = CLIPModel.from_pretrained(base_model_path)
    num_classes = 3
    model = CLIPImageClassifier(base_model, num_classes)
    model.load_state_dict(torch.load(model_path))
    clip_zeroshot = False

    data_loader = data_preparation(config)

    y_true, y_pred, class_names, model = eval_model(
        processor, model, data_loader, clip_zeroshot
    )

    metrics(y_true, y_pred, class_names, mlflow_experiment_name, mlflow_run_name)
