from src.model import AutoSeg
from src.model import sequence_contrastive_loss
from src.test_dataset import PascalVOCDataset, collate_fn_factory
from torch.utils.data import DataLoader
from transformers import CLIPProcessor, CLIPModel
from torch.optim import Adam
from torch import nn
import torch
import tqdm
import math

# labels - index dict
labels = [
    "unlabeled",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "potted plant",
    "sheep",
    "sofa",
    "train",
    "tv/monitor",
]


def mean_iou(true_labels, pred_labels, num_classes):
    """
    Calculate the mean Intersection over Union (mIoU) score.

    :param true_labels: array of shape (N, H, W), true labels
    :param pred_labels: array of shape (N, H, W), predicted labels
    :param num_classes: integer, number of classes
    :return: float, mean IoU score
    """
    iou_list = []
    for cls in range(num_classes):
        true_positive = ((pred_labels == cls) & (true_labels == cls)).sum()
        false_positive = ((pred_labels == cls) & (true_labels != cls)).sum()
        false_negative = ((pred_labels != cls) & (true_labels == cls)).sum()
        intersection = true_positive
        union = true_positive + false_positive + false_negative
        if union == 0:
            iou = float("nan")
        else:
            iou = intersection / union
        iou_list.append(iou)

    iou_list = [x for x in iou_list if not math.isnan(x)]

    # Compute the mean IoU across all classes
    mIoU = sum(iou_list) / len(iou_list)
    return mIoU


def harmonic_mean_iou(mIoU_seen, mIoU_unseen):
    """
    Calculate the harmonic mean of the mean IoU (Intersection over Union) for seen and unseen classes.

    :param mIoU_seen: float, mean IoU for seen classes
    :param mIoU_unseen: float, mean IoU for unseen classes
    :return: float, harmonic mean IoU
    """
    # Ensure both mIoU_seen and mIoU_unseen are non-zero to avoid division by zero.
    if mIoU_seen > 0 and mIoU_unseen > 0:
        return (2 * mIoU_seen * mIoU_unseen) / (mIoU_seen + mIoU_unseen)
    else:
        return 0
    
def binary_masks_to_label_mask(binary_masks, output_labels):
    N, L, W, H = binary_masks.shape
    label_masks = torch.zeros((N, W, H), dtype=torch.int64)

    for i, label in enumerate(labels):
        index = output_labels.index(label)
        label_masks[binary_masks[:, index, :, :].bool()] = i
    
    return label_masks

device = 0

# Define dataset dir
# change to the path of the VOC2012 dataset
dataset_dir = "datasets/VOCdevkit/VOC2012/"
image_dir = "JPEGImages/"
label_file_path = "src/test_labels.txt"
data_file_path = "ImageSets/Segmentation/val.txt"
annotation_dir = "SegmentationClass/"


# Create dataset object
data = PascalVOCDataset(
    dataset_dir + image_dir,
    dataset_dir + annotation_dir,
    label_file_path,
    dataset_dir + data_file_path,
)

processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

collate_fn = collate_fn_factory(processor)

data_loader = DataLoader(data, batch_size=32, collate_fn=collate_fn, num_workers=0)

# Step 2: Define the model architecture
model = AutoSeg()
model.load_state_dict(torch.load("/scratch/t.tovi/models/autoseg_v0.1"))
model.eval()

for batch in tqdm.tqdm(data_loader):
    pixel_values = batch["pixel_values"].to(device)
    masks = batch["masks"].to(device)
    input_ids = batch["input_ids"].to(device)
    token_summary_idx = batch["token_summary_idx"].to(device)

    source_ids = input_ids[:, :-1]
    token_summary_idx = token_summary_idx[:, :-1]

    target_labels = model.text_encoder.embeddings(input_ids[:, 1:])
    target_masks = masks[:, 1:]

    # note: one mask for each of the labels output by the model
    output_masks, output_labels = model(
        pixel_values, source_ids, token_summary_idx=token_summary_idx
    )

    mixed_output_masks = binary_masks_to_label_mask(output_masks)
    mixed_target_masks = binary_masks_to_label_mask(target_masks)

    mean_iou_score = mean_iou(mixed_target_masks, mixed_output_masks, num_classes=21)

    print(f"Mean IoU: {mean_iou_score}")


