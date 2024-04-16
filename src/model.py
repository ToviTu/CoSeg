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
        super().__init__(nhead=4, nencoder=4, ndecoder=4)

        vision_model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
        text_model = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")

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
            d_model = d_text,
            nhead = nhead,
            num_encoder_layers = nencoder,
            num_decoder_layers = ndecoder,
            activation = nn.GELU(),
            batch_first = True
        )

        # Projection
        self.label_head = nn.Linear(d_text, d_text)

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
        return self.text_model.embeddings(input_ids)

    def visual_forward(self, pixel_values, output_hidden_states=False):
        # image -> text_embeddings
        return self.vision_encoder(pixel_values, output_hidden_states=output_hidden_states)

    def text_forward(self, text_embeddings):
        # text_embedding -> text_embeddings
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
        # image X index -> text_embeddings

        # Get visual features
        visual_outputs = self.visual_forward(pixel_values, output_hidden_states=output_hidden_states)
        visual_features = self.vision_projector(visual_outputs.last_hidden_state)

        # Get text embeddings
        text_outputs = self.text_model(input_ids).last_hidden_state
        text_features = self.text_projector(text_outputs)

        # Decode texts
        text_pred = self.decoder_forward(visual_features, text_features)

        assert input_ids.shape[1] == text_pred.shape[1]

        # Return both text embeddings and visual activations
        if output_hidden_states:
            return text_pred, [visual_outputs.hidden_states[i] for i in (3, 6, 9)]

        # Get text embeddings
        return text_pred

    def forward(self, pixel_values, input_ids, output_hidden_states = False):

        if output_hidden_states:
            text_pred, hidden_states = self.cond_forward(pixel_values, input_ids, output_hidden_states=output_hidden_states)
            return self.label_head(text_pred), hidden_states

        return self.label_head(self.cond_forward(pixel_values, input_ids))

class CLIPLang_prefix(nn.Module):

    '''
    This class is for detecting most salient object
    in the scene as an auto-regressive task
    '''

    def __init__(self, nhead=4, nencoder=4, clip_version="openai/clip-vit-base-patch16"):
        super().__init__()

        vision_model = CLIPVisionModelWithProjection.from_pretrained(clip_version)
        text_model = CLIPTextModelWithProjection.from_pretrained(clip_version)

        # vision model
        self.vision_encoder = vision_model.vision_model
        self.vision_projector = vision_model.visual_projection

        #text model
        self.text_model = text_model.text_model
        self.text_projector = text_model.text_projection

        # internal dimensions
        self.d_text = self.text_model.embeddings.token_embedding.weight.shape[1]
        self.d_image = self.vision_encoder.embeddings.position_embedding.weight.shape[1]

        # Learnable embeddings
        self.query_embeddings = nn.Embedding(20, self.d_text)
        self.query_embeddings.weight.data.normal_(mean=0, std=0.02)

        # transformer for next label prediction
        encoder_layers = nn.TransformerEncoderLayer(
            d_model = self.d_text,
            nhead = nhead,
            activation = nn.GELU(),
            batch_first = True
        )
        self.decoder = nn.TransformerEncoder(encoder_layers, num_layers=nencoder)

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
        return self.text_model.embeddings(input_ids)
    
    def position_encode(self, embeddings):
        index_tensor = torch.arange(embeddings.shape[1]).repeat(embeddings.shape[0], 1).to(embeddings.device)
        return embeddings + self.text_model.embeddings.position_embedding(index_tensor)

    def visual_forward(self, pixel_values, output_hidden_states=False):
        # image -> text_embeddings
        return self.vision_encoder(pixel_values, output_hidden_states=output_hidden_states)

    def text_forward(self, text_embeddings):
        # text_embedding -> text_embeddings
        return self.text_model.encoder(text_embeddings)

    def decoder_forward(self, img_src, txt_tgt):
        # Send image seq and text seq to lang model
        # Concatenate prefix and visual embeddings
        decoder_input = torch.cat((txt_tgt, img_src), dim=1)
        # Return only the query tokens
        return self.decoder(decoder_input)[:, :txt_tgt.shape[1], :]

    def cond_forward(self, pixel_values, output_hidden_states = False):
        # Get visual features
        visual_outputs = self.visual_forward(pixel_values, output_hidden_states=output_hidden_states)
        visual_features = self.vision_projector(visual_outputs.last_hidden_state)

        # Get text embeddings
        # text_outputs = self.text_model.encoder(embeddings).last_hidden_state
        # text_features = self.text_projector(text_outputs)

        # Remove the [cls] token
        visual_features = visual_features[:, 1:, :]

        # Initialize query tokens
        index_tensor = torch.arange(20).repeat(pixel_values.shape[0], 1).to(pixel_values.device)
        query_tokens = self.query_embeddings(index_tensor)
        query_tokens = self.position_encode(query_tokens)

        # Decode texts
        text_pred = self.decoder_forward(visual_features, query_tokens)

        # Return both text embeddings and visual activations
        if output_hidden_states:
            return text_pred, [visual_outputs.hidden_states[i] for i in (3, 6, 9)]

        # Get text embeddings
        return text_pred

    def forward(self, pixel_values, output_hidden_states = False):

        if output_hidden_states:
            text_pred, hidden_states = self.cond_forward(pixel_values, output_hidden_states=output_hidden_states)
            return self.label_head(text_pred), hidden_states

        return self.label_head(self.cond_forward(pixel_values, embeddings))

class CLIPLang_xatten(nn.Module):

    '''
    This class is for detecting most salient object
    in the scene as an auto-regressive task
    '''

    def __init__(self, nhead=4, nencoder=4, ndecoder=4, clip_version="openai/clip-vit-base-patch16"):
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

        # Learnable embeddings
        self.query_embeddings = nn.Embedding(20, self.d_text)
        self.query_embeddings.weight.data.normal_(mean=0, std=0.02)

        # transformer for next label prediction
        self.decoder = nn.Transformer(
            d_model = self.d_text,
            nhead = nhead,
            num_encoder_layers = nencoder,
            num_decoder_layers = ndecoder,
            activation = nn.GELU(),
            batch_first = True
        )


    def to_embedding(self, input_ids):
        # index -> text_embeddings
        return self.text_model.embeddings.token_embedding(input_ids)

    def position_encode(self, embeddings):
        index_tensor = torch.arange(embeddings.shape[1]).repeat(embeddings.shape[0], 1).to(embeddings.device)
        return embeddings + self.text_model.embeddings.position_embedding(index_tensor)

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
        )

    def cond_forward(self, pixel_values, output_hidden_states = False):
        # Get visual features
        visual_outputs = self.visual_forward(pixel_values, output_hidden_states=output_hidden_states)
        visual_features = self.vision_projector(visual_outputs.last_hidden_state)

        # Remove the [cls] token
        visual_features = visual_features[:, 1:, :]

        # Initialize query tokens
        index_tensor = torch.arange(20).repeat(pixel_values.shape[0], 1).to(pixel_values.device)
        query_tokens = self.query_embeddings(index_tensor)
        query_tokens = self.position_encode(query_tokens)

        # Decode texts
        text_pred = self.decoder_forward(visual_features, query_tokens)

        # Return both text embeddings and visual activations
        if output_hidden_states:
            return text_pred, [visual_outputs.hidden_states[i] for i in (3, 6, 9)]

        # Get text embeddings
        return text_pred

    def forward(self, pixel_values, output_hidden_states = False):

        if output_hidden_states:
            text_pred, hidden_states = self.cond_forward(pixel_values, output_hidden_states=output_hidden_states)
            return text_pred, hidden_states

        return self.cond_forward(pixel_values)

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
        




