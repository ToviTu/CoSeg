from src.model import *
from src.dataset import COCOStuffDataset, collate_fn_factory
from torch.utils.data import DataLoader
from transformers import CLIPProcessor, CLIPModel
from torch.optim import Adam
from torch import nn
import torch
import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import CLIPProcessor, CLIPModel, CLIPTextModelWithProjection, CLIPVisionModelWithProjection
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as TF
import os
import numpy as np

device = 0

# Define dataset dir
dataset_dir = "/scratch/t.tovi/datasets/"

# Create dataset object
data = COCOStuffDataset(
    dataset_dir+"COCO_stuff_images/train2017", 
    dataset_dir+"COCO_stuff_annotations/train2017"
)

# Load CLIP processor
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Define collate function
collate_fn = collate_fn_factory(processor)

# Create batch data loader
data_loader = DataLoader(data, batch_size=32, collate_fn=collate_fn, num_workers=2)

# Initialize the model
model = AutoSeg()
m = nn.Sigmoid()
model.to(device)
model.requires_grad_(False)

# # Unfreeze decoders
encoder_params = [
    model.encoders.decoder,
    model.encoders.label_head,
]

decoder_params = [
    model.reduces,
    model.film_mul,
    model.film_add,
    model.decoder,
    model.mask_head
]

for param in encoder_params + decoder_params:
    param.requires_grad_(True)

# Define training parameters
lr_encoder = 1e-4
lr_decoer = 1e-3
alpha = 0.08
num_epochs = 5

# Optimizer
optim = Adam(
    [
        {'params': param.parameters(), "lr" : lr_encoder}
        for param in encoder_params
    ] +\
    [
        {'params': param.parameters(), "lr" : lr_decoer}
        for param in decoder_params
    ]
)

# Loss
SC = sequence_contrastive_loss
BCE = nn.BCELoss()
CE = nn.CrossEntropyLoss()

count = 0
for _ in range(num_epochs):
    for batch in data_loader:
        # Prepare data
        pixel_values = batch['pixel_values'].to(device)
        masks = batch['masks'].to(device)
        input_ids = batch['input_ids'].to(device)
        token_summary_idx = batch['token_summary_idx'].to(device)

        # Prepare input and output

        input = input_ids[:, :-1]
        target = input_ids[:, 1:]

        mask_logits, label_logits = model(pixel_values, input, token_summary_idx)

        # Compute loss
        l1 = BCE(m(mask_logits), masks[:, 1:])
        l2 = alpha * CE(label_logits.permute(0, 2, 1), target)
        loss = l1 + l2

        loss.backward()
        optim.step()
        optim.zero_grad()

        if count % 100 == 0:
            print(f"Last batch loss: {loss.detach().cpu().item()}, {l1.detach().cpu().item()}, {l2.detach().cpu().item()}")
        count += 1

        # if count % -1 == 0:
        #     break
    print("One training epoch done")
    torch.save(model.state_dict(), "/scratch/t.tovi/autoseg_v0.1")