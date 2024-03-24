from src.model import *
from src.dataset import COCOStuffDataset, collate_fn_factory
from torch.utils.data import DataLoader
from transformers import CLIPProcessor, CLIPModel
from torch.optim import Adam
from torch import nn
import torch
import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR

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

# Unfreeze decoders
model.encoders.decoder.requires_grad_(True)
model.film_mul.requires_grad_(True)
model.film_add.requires_grad_(True)
model.autoseg_decoder.requires_grad_(True)


# Define training parameters
lr_max = 1e-5
lr_min = 1e-8
alpha = 0.01
num_epochs = 1

# Optimizer
optim = Adam(
    [
        {'params': model.encoders.decoder.parameters()},
        {'params': model.film_mul.parameters()},
        {'params': model.film_add.parameters()},
        {'params': model.autoseg_decoder.parameters()}
    ],
    lr = lr_max
)
#scheduler = CosineAnnealingLR(optim, T_max=200, eta_min=lr_min)

# Loss 
loss_1 = sequence_contrastive_loss
loss_2 = nn.BCELoss()

count = 0
for _ in range(num_epochs):
    for batch in tqdm.tqdm(data_loader):
        # Prepare data
        pixel_values = batch['pixel_values'].to(device)
        masks = batch['masks'].to(device)
        input_ids = batch['input_ids'].to(device)
        token_summary_idx = batch['token_summary_idx'].to(device)

        # Get text embeddings
        text_embeddings = model.encoders.to_embedding(input_ids, token_summary_idx)
        text_embeddings = text_embeddings.detach()

        # Shift the sequence
        input_embeddings = text_embeddings[:, :-1]
        target_embeddings = text_embeddings[:, 1:]

        pred_masks, pred_embeddings = model(pixel_values, input_embeddings)

        # Compute loss
        l1 = loss_1(pred_embeddings, target_embeddings)
        l2 = loss_2(m(pred_masks), masks[:,1:1+input_embeddings.shape[1],:,:])

        loss = alpha * l1 + (1 - alpha) * l2
            
        loss.backward()
        optim.step()
        optim.zero_grad()

        if count % 100 == 0:
            print(f"Last batch loss: {loss.detach().cpu().item()}")
            print(f"Last batch loss1: {l1.detach().cpu().item()}")
            print(f"Last batch loss2: {l2.detach().cpu().item()}")
            #scheduler.step()
        count += 1
    print("One training epoch done")
    torch.save(model.state_dict(), "/scratch/t.tovi/models/autoseg_v0.1")