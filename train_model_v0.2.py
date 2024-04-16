import torch
from torch import nn
from transformers import CLIPProcessor, CLIPModel, CLIPTextModelWithProjection, CLIPVisionModelWithProjection
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as TF
import os
import numpy as np
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
import tqdm

"""## Dataset"""

src_dir = "/home/research/jianhong.t/OpenVocab_Seg_with_AutoRegres/src/"
dataset_dir = "/scratch/t.tovi/datasets/"
image_dir = "COCO_stuff_images/train2017/"
annotation_dir = "COCO_stuff_annotations/train2017/"

class COCOStuffDataset(Dataset):
    def __init__(self, image_dir, annotation_dir, img_size=224):
        """
        Args:
            image_dir (string): Directory with all the images.
            annotation_dir (string): Directory with all the annotations.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.images = os.listdir(image_dir)
        self.img_size = img_size

        # Load the label mapping
        self.digit_to_object_mapping = {}
        with open(f'{src_dir}labels.txt', 'r') as file:
            for line in file:
                key, value = line.strip().split(':')
                self.digit_to_object_mapping[int(key)] = value.strip()
        self.digit_to_object_mapping[255] = "unlabled"

    def center_crop(self, image, mask):
        transform = transforms.CenterCrop(self.img_size)
        return transform(image), transform(mask)

    def resize(self, image, mask):
        transform = transforms.CenterCrop(min(image.size[1:]))

        cropped_image = transform(image)
        cropped_mask = transform(mask)

        resized_image = transforms.Resize((self.img_size, self.img_size),transforms.InterpolationMode.BILINEAR)(cropped_image)
        resized_mask= transforms.Resize((self.img_size, self.img_size), transforms.InterpolationMode.NEAREST)(cropped_mask)
        return resized_image, resized_mask

    def __len__(self):
        return len(self.images)

    def get(self, idx):
        # Load image
        img_name = os.path.join(self.image_dir, self.images[idx])
        image = Image.open(img_name)#.convert('RGB')

        annotation_name = os.path.join(self.annotation_dir, self.images[idx].replace('.jpg', '.png'))
        annotation = Image.open(annotation_name)

        ids = np.unique(np.array(annotation))
        labels = [self.digit_to_object_mapping[id] for id in ids]

        return image, annotation, labels

    def __getitem__(self, idx):
        # Load image
        img_name = os.path.join(self.image_dir, self.images[idx])
        image = Image.open(img_name)#.convert('RGB')

        # Load annotation
        annotation_name = os.path.join(self.annotation_dir, self.images[idx].replace('.jpg', '.png'))
        annotation = Image.open(annotation_name)

        image, mask = self.resize(image, annotation)
        mask = np.array(mask)
        mask += 1
        mask[mask==256] = 0


        # Indexed masks
        ids = np.unique(mask)
        ids = [id for id in ids if id != 0]
        nonempty_masks = [np.full(mask.shape, id) * (mask==id) for id in ids]
        nonempty_masks = sorted(nonempty_masks, key=lambda x: np.sum(x!=0), reverse=True)

        # Get ids and labels
        ids = [np.unique(mask)[-1] for mask in nonempty_masks]
        labels = [self.digit_to_object_mapping[id] for id in ids]

        # Convert to binary masks
        nonempty_masks = [(mask != 0).astype(float) for mask in nonempty_masks]

        sample = {'image': image, 'annotation': nonempty_masks, 'labels': labels}

        return sample

"""## Label Model"""

class CLIPLang(nn.Module):

    '''
    This class is for detecting most salient object
    in the scene as an auto-regressive task
    '''

    def __init__(self, nhead=4, nencoder=3, ndecoder=6, clip_version="openai/clip-vit-base-patch16"):
        super().__init__()

        vision_model = CLIPVisionModelWithProjection.from_pretrained(clip_version)
        text_model = CLIPTextModelWithProjection.from_pretrained(clip_version)

        # vision model
        self.vision_encoder = vision_model.vision_model
        self.vision_projector = vision_model.visual_projection

        # text model
        self.text_model = text_model.text_model
        self.text_projector = text_model.text_projection

        # internal dimensions
        self.d_text = self.text_model.embeddings.token_embedding.weight.shape[1]
        self.d_image = self.vision_encoder.embeddings.position_embedding.weight.shape[1]

        # transformer for next label prediction
        self.decoder = nn.Transformer(
            d_model = self.d_text,
            nhead = nhead,
            num_encoder_layers = nencoder,
            num_decoder_layers = ndecoder,
            activation = nn.GELU(),
            batch_first = True
        )

        # Projection
        self.label_head = nn.Linear(self.d_text, self.d_text)

        self.EOS_EMBEDDING = self.text_model.embeddings(torch.tensor([49407]))[0, 0, :]

    def sum_batch_entity_embeddings(self, batch_embeddings, batch_labels):
        output_seq_length = torch.max(batch_labels+1)
        summed_embeddings = torch.zeros((
            batch_embeddings.shape[0],
            output_seq_length,
            batch_embeddings.shape[-1]
        )).to(batch_embeddings.device)
        for i in range(batch_embeddings.shape[0]):
            for j in range(batch_embeddings.shape[1]):
                summed_embeddings[i, batch_labels[i, j]] += batch_embeddings[i, j]

            for k in range(torch.max(batch_labels[i]), output_seq_length):
                assert torch.zeros((1,2)).sum() == 0, "Error in summing embeddings"
                summed_embeddings[i, k] += self.EOS_EMBEDDING.to(batch_embeddings.device)
        summed_embeddings = summed_embeddings.detach()
        return summed_embeddings

    def to_embedding(self, input_ids):
        # index -> text_embeddings
        return self.text_model.embeddings.token_embedding(input_ids)

    def visual_forward(self, pixel_values, output_hidden_states=False):
        # image -> text_embeddings
        return self.vision_encoder(pixel_values, output_hidden_states=output_hidden_states)

    def text_forward(self, text_embeddings):
        # text_embedding -> text_embeddings
        return self.text_model.encoder(text_embeddings)

    def decoder_forward(self, img_src, txt_tgt):
        # Send image seq and text seq to lang model
        return self.decoder(
            img_src,
            txt_tgt,
            tgt_mask=self.decoder.generate_square_subsequent_mask(
                txt_tgt.shape[1],
                device=txt_tgt.device,
            ),
            tgt_is_causal=True
        )

    def cond_forward(self, pixel_values, embeddings, output_hidden_states = False):
        # Get visual features
        visual_outputs = self.visual_forward(pixel_values, output_hidden_states=output_hidden_states)
        visual_features = self.vision_projector(visual_outputs.last_hidden_state)

        # Get text embeddings
        text_outputs = self.text_model.encoder(embeddings).last_hidden_state
        text_features = self.text_projector(text_outputs)

        # Decode texts
        text_pred = self.decoder_forward(visual_features, text_features)

        assert embeddings.shape[1] == text_pred.shape[1]

        # Return both text embeddings and visual activations
        if output_hidden_states:
            return text_pred, [visual_outputs.hidden_states[i] for i in (3, 6, 9)]

        # Get text embeddings
        return text_pred

    def forward(self, pixel_values, embeddings, output_hidden_states = False):

        if output_hidden_states:
            text_pred, hidden_states = self.cond_forward(pixel_values, embeddings, output_hidden_states=output_hidden_states)
            return self.label_head(text_pred), hidden_states

        return self.label_head(self.cond_forward(pixel_values, embeddings))

"""## Segmentation Model"""

class AutoSeg(nn.Module):
    def __init__(self, d_reduce=64, nhead=4, nencoder=3, ndecoder=6):
        super().__init__()

        self.encoders = CLIPLang(nhead=nhead, nencoder=nencoder, ndecoder=ndecoder)
        self.reduces = nn.ModuleList([
            nn.Linear(self.encoders.d_image, d_reduce) for _ in range(3)
        ])
        self.film_mul = nn.Linear(self.encoders.d_text, d_reduce)
        self.film_add = nn.Linear(self.encoders.d_text, d_reduce)

        self.decoder = nn.ModuleList([
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=d_reduce,
                    nhead=nhead,
                    dim_feedforward=2048,
                    dropout=0.1,
                    activation=nn.GELU(),
                    batch_first=True,
                ),
                num_layers=1,
            )
            for _ in range(3)
        ])

        self.mask_head = nn.Sequential(
            nn.Conv2d(d_reduce, d_reduce, kernel_size=(3, 3), padding=(1, 1), padding_mode="replicate"),
            nn.GELU(),
            nn.ConvTranspose2d(d_reduce, d_reduce//2, kernel_size=(4, 4), stride=(4, 4)),
            nn.GELU(),
            nn.ConvTranspose2d(d_reduce//2, 1, kernel_size=(4, 4), stride=(4, 4))
        )

    def forward(self, pixel_values, text_values):
        # Get text embeddings
        lang_output, hidden_states = self.encoders.cond_forward(pixel_values, text_values, output_hidden_states=True)

        # Image sequence size
        self.image_seq_size = int(np.sqrt(hidden_states[0].shape[1]))

        masks = []
        for i, batch_embeddings in enumerate(lang_output.permute(1, 0, 2)):
            a  = None
            for hs, block, reduce in zip(hidden_states, self.decoder, self.reduces):
                hs = hs.permute(1, 0, 2)
                if a is None:
                    a = reduce(hs)
                else:
                    a = a + reduce(hs)

                a = a * self.film_mul(batch_embeddings) + self.film_add(batch_embeddings)
                a = block(a)

            a = a[1:].permute(1, 2, 0)
            a = a.view(a.shape[0], a.shape[1], self.image_seq_size, self.image_seq_size)
            a = self.mask_head(a)
            masks.append(a)

        masks = torch.cat(masks, dim=1)
        return masks, self.encoders.label_head(lang_output)

"""## Define the collate function"""

def collate_fn_factory(processor, eos_embedding, bos_embedding, label_embeddings):

    def collate_fn(batch):
        size = processor.image_processor.size['shortest_edge'] #224
        transform = transforms.ToTensor()

        # Preprocess pixel values
        images = [each['image'] for each in batch]
        batch_pixel_values = processor(None, images=images, return_tensors='pt')['pixel_values']

        # Preprocess texts
        batch_labels = [each['labels'] for each in batch]
        batch_lookup_ids = [[reverse_mapping[label] for label in labels]for labels in batch_labels]
        sizes = [len(each) for each in batch_lookup_ids] # number of labels
        max_size = max(sizes) + 2 # max number of labels + 2 embeddings

        # Create batch_embeddings
        batch_embeddings = eos_embedding.repeat(len(batch), max_size, 1).clone()
        batch_embeddings[:, 0, :] = bos_embedding
        for i, each in enumerate(batch_lookup_ids):
            batch_embeddings[i, 1:sizes[i]+1] = label_embeddings[each]

        # Preprocess masks
        batch_masks = np.stack([
            np.stack([np.zeros((size, size))] + each['annotation'] + [np.zeros((size, size))] * (max_size - len(each['annotation']) - 1) )
            for each in batch
        ])
        batch_masks = torch.tensor(batch_masks)

        return {
            "pixel_values": batch_pixel_values,
            "masks": batch_masks.type(torch.float32),
            "embeddings": batch_embeddings.type(torch.float32)
        }

    return collate_fn

"""## Define the CLIP loss"""

def clip_loss(pred_features, target_features, temperature=0.07):
    """
    Compute the CLIP loss between image and text features.

    Parameters:
    - image_features: A tensor of shape (batch_size, feature_dim) containing the image features.
    - text_features: A tensor of shape (batch_size, feature_dim) containing the text features.
    - temperature: A scalar temperature parameter.

    Returns:
    - The CLIP loss.
    """
    # Normalize features
    pred_features = F.normalize(pred_features, dim=-1)
    target_features = F.normalize(target_features, dim=-1)

    # Compute similarity matrix
    similarity = torch.matmul(pred_features, target_features.T) / temperature

    # Image-to-text and text-to-image loss
    loss = F.cross_entropy(similarity, torch.arange(len(pred_features)).to(pred_features.device))

    # Symmetric loss
    return loss

"""## Training Pipeline"""

device = 0

# Define dataset dir
dataset_dir = "/scratch/t.tovi/datasets/"

# Create dataset object
data = COCOStuffDataset(
    dataset_dir+"COCO_stuff_images/train2017",
    dataset_dir+"COCO_stuff_annotations/train2017",
    img_size=224
)

lang_model = CLIPLang(nhead=4, nencoder=4, ndecoder=4)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

# Get loss query table

labels = data.digit_to_object_mapping

label_indices = list(data.digit_to_object_mapping.keys())
label_text = [data.digit_to_object_mapping[each] for each in label_indices]
label_indices = processor(label_text, padding=True, return_tensors='pt')['input_ids']

with torch.no_grad():
    label_embeddings = lang_model.text_model(label_indices)["pooler_output"]
    eos_embedding = lang_model.text_model(torch.tensor([processor.tokenizer.eos_token_id]))["pooler_output"]
    bos_embedding = lang_model.text_model(torch.tensor([processor.tokenizer.bos_token_id]))["pooler_output"]

label_embeddings.requires_grad_(False)
eos_embedding.requires_grad_(False)
bos_embedding.requires_grad_(False)

reverse_mapping = {v: k for k, v in data.digit_to_object_mapping.items()}

# Get the collate function

collate_fn = collate_fn_factory(processor, eos_embedding, bos_embedding, label_embeddings)

# Create batch data loader
data_loader = DataLoader(data, batch_size=16, collate_fn=collate_fn, num_workers=4, shuffle=True)

# Initialize the model
model = AutoSeg(d_reduce=64)
m = nn.Sigmoid()
model.to(device)

# Freeze all
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
lr_decoer = 1e-4
alpha = 0.025  
beta = 1
num_epochs = 7

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

scheduler = CosineAnnealingLR(optim, T_max=len(data_loader), eta_min=1e-6)

# Loss
mask_objective = nn.BCELoss()
lang_objective = clip_loss

"""## Train"""

#from IPython.display import clear_output

count = 0
for _ in range(num_epochs):
    batch_loss = 0
    batch_l1 = 0
    batch_l2 = 0
    for batch in data_loader:
        # Prepare data
        pixel_values = batch['pixel_values'].to(device)
        masks = batch['masks'].to(device)
        embeddings = batch['embeddings'].to(device)

        # Prepare input and output

        input = embeddings[:, :-1]
        target = embeddings[:, 1:]

        mask_logits, label_embeddings = model(pixel_values, input)

        # Compute loss
        mask_prob = m(mask_logits)
        l1 = mask_objective(mask_prob, masks[:, 1:])
        l2 = alpha * lang_objective(target.flatten(start_dim=0, end_dim=1), label_embeddings.flatten(start_dim=0, end_dim=1))

        # Total loss
        loss = l1 + l2

        loss.backward()
        optim.step()
        optim.zero_grad()

        batch_loss += loss.detach().cpu().item()
        batch_l1 += l1.detach().cpu().item()
        batch_l2 += l2.detach().cpu().item()

        scheduler.step()

        if (count+1) % 64 == 0:
            print(f"Avrage batch loss: {batch_loss / 64}, {batch_l1 / 64}, {batch_l2 / 64}")
            batch_loss = 0
            batch_l1 = 0
            batch_l2 = 0

        count += 1


    print("One training epoch done")
    torch.save(model.state_dict(), "/scratch/t.tovi/autoseg_v0.1")
