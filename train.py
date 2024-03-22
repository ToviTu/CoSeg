from src.model import AutoSeg
from src.model import sequence_contrastive_loss
from src.dataset import COCOStuffDataset, collate_fn_factory
from torch.utils.data import DataLoader
from transformers import CLIPProcessor, CLIPModel
from torch.optim import Adam
from torch import nn
import tqdm

device = 0

# Define dataset dir
dataset_dir = "/scratch/t.tovi/datasets/"

# Create dataset object
data = COCOStuffDataset(
    dataset_dir + "COCO_stuff_images/train2017",
    dataset_dir + "COCO_stuff_annotations/train2017",
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

# Define training parameters
lr = 1e-4
alpha = 0.3

# Optimizer
optim = Adam(model.autoseg_decoder.parameters(), lr=lr)

# Loss
loss_1 = sequence_contrastive_loss
loss_2 = nn.BCELoss()

count = 0
for batch in tqdm.tqdm(data_loader):
    pixel_values = batch["pixel_values"].to(device)
    masks = batch["masks"].to(device)
    input_ids = batch["input_ids"].to(device)
    token_summary_idx = batch["token_summary_idx"].to(device)

    source_ids = input_ids[:, :-1]
    token_summary_idx = token_summary_idx[:, :-1]

    target_labels = model.text_encoder.embeddings(input_ids[:, 1:])
    target_masks = masks[:, 1:]

    output_masks, output_labels = model(
        pixel_values, source_ids, token_summary_idx=token_summary_idx
    )

    loss = loss_2(
        m(output_masks), target_masks[:, : output_masks.shape[1], :, :]
    ) + alpha * loss_1(output_labels, target_labels)
    loss.backward()
    optim.step()
    optim.zero_grad()

    if count % 100 == 0:
        print(f"Last batch loss: {loss.detach().cpu().item()}")
    count += 1

torch.save(model.state_dict(), "/scratch/t.tovi/models/autoseg_v0.1")
