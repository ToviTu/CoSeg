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

class CLIPLang(nn.Module):

    '''
    This class is for detecting most salient object
    in the scene as an auto-regressive task
    '''

    def __init__(self):
        super().__init__()

        vision_model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
        text_model = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")

        # vision model
        self.vision_encoder = vision_model.vision_model
        self.vision_projector = vision_model.visual_projection

        # text model
        self.text_model = text_model.text_model
        self.text_projector = text_model.text_projection

        # transformer for next label prediction
        self.decoder = nn.Transformer(
            d_model = 512,
            nhead = 4,
            num_encoder_layers = 3,
            num_decoder_layers = 6,
            activation = nn.GELU(),
            batch_first = True
        )
        self.label_head = nn.Linear(512,  self.text_model.embeddings.token_embedding.num_embeddings)

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
        return self.text_model.embeddings(input_ids)

    def visual_forward(self, pixel_values, output_hidden_states=False):

        return self.vision_encoder(pixel_values, output_hidden_states=output_hidden_states)

    def text_forward(self, text_embeddings):
        return self.text_model(text_embeddings)

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

    def cond_forward(self, pixel_values, input_ids, output_hidden_states = False):

        # Get visual features
        visual_outputs = self.visual_forward(pixel_values, output_hidden_states=output_hidden_states)

        visual_features = self.vision_projector(visual_outputs.last_hidden_state)

        # Get text embeddings
        text_outputs = self.text_model(input_ids).last_hidden_state
        text_features = self.text_projector(text_outputs)

        # Decode texts
        text_pred = self.decoder_forward(visual_features, text_features)

        assert input_ids.shape[1] == text_pred.shape[1]

        if output_hidden_states:
            return text_pred, [visual_outputs.hidden_states[i] for i in (3, 6, 9)]

        # Get token prediction
        return text_pred

    def forward(self, pixel_values, input_ids, output_hidden_states = False):

        if output_hidden_states:
            text_pred, hidden_states = self.cond_forward(pixel_values, input_ids, output_hidden_states=output_hidden_states)
            return self.label_head(text_pred), hidden_states

        return self.label_head(self.cond_forward(pixel_values, input_ids))


class AutoSeg(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoders = CLIPLang()
        self.reduces = nn.ModuleList([
            nn.Linear(768, 64) for _ in range(3)
        ])
        self.film_mul = nn.Linear(512, 64)
        self.film_add = nn.Linear(512, 64)

        self.decoder = nn.ModuleList([
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=64,
                    nhead=4,
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
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.GELU(),
            nn.ConvTranspose2d(64, 32, kernel_size=(4, 4), stride=(4, 4)),
            nn.GELU(),
            nn.ConvTranspose2d(32, 1, kernel_size=(8, 8), stride=(8, 8)),
        )

    def forward(self, pixel_values, text_values, token_summary_idx):
        # Get text embeddings
        lang_output, hidden_states = self.encoders.cond_forward(pixel_values, text_values, output_hidden_states=True)
        agg_lang_output = self.encoders.sum_batch_entity_embeddings(lang_output, token_summary_idx)

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
            a = a.view(a.shape[0], a.shape[1], 7, 7)
            a = self.mask_head(a)
            masks.append(a)


        masks = torch.cat(masks, dim=1)
        return masks, self.encoders.label_head(lang_output)
        

def sequence_contrastive_loss(seq1_features, seq2_features, temperature=0.07):
    """
    Compute a contrastive loss between two sequences of token embeddings, aiming to maximize
    the cosine similarity of corresponding tokens and minimize it for non-corresponding ones.

    Parameters:
    - seq1_features: tensor of shape (N, L, D), embeddings for sequence 1.
    - seq2_features: tensor of shape (N, L, D), embeddings for sequence 2.
    - temperature: a scalar, temperature parameter.

    Returns:
    - A scalar tensor with the contrastive loss.
    """
    N, L, D = seq1_features.size()  # Batch size, sequence length, feature dimension
    seq1_features = nn.functional.normalize(seq1_features, p=2, dim=-1)
    seq2_features = nn.functional.normalize(seq2_features, p=2, dim=-1)

    # Reshape to (N*L, D) to treat each token as a separate data point
    seq1_features_flat = seq1_features.view(-1, D)
    seq2_features_flat = seq2_features.view(-1, D)

    # Similarity matrix
    similarity = torch.matmul(seq1_features_flat, seq2_features_flat.T) / temperature

    # Create labels
    labels = torch.arange(N*L).to(similarity.device)

    # Compute the loss
    loss = nn.functional.cross_entropy(similarity, labels)

    return loss        
        




