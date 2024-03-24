import torch
from torch import nn
from transformers import CLIPProcessor, CLIPModel
import numpy as np
import torch.nn.functional as F

class CLIPLang(nn.Module):

    '''
    This class is for detecting most salient object
    in the scene as an auto-regressive task
    '''

    def __init__(self, clip_encoders=None):
        super().__init__()

        CLIP = CLIPModel.from_pretrained("openai/clip-vit-base-patch32") \
            if clip_encoders is None else clip_encoders

        # vision model
        self.vision_encoder = CLIP.vision_model
        self.vision_projector = CLIP.visual_projection

        # text model
        self.embeddings = CLIP.text_model.embeddings
        self.text_encoder = CLIP.text_model.encoder
        self.text_projector = CLIP.text_projection

        # transformer for next label prediction
        self.decoder = nn.Transformer(
            d_model = 512,
            nhead = 8,
            num_encoder_layers = 6,
            num_decoder_layers = 6,
            activation = nn.GELU(),
            batch_first = True
        )
        self.label_head = nn.Linear(512, 512)

        self.EOS_EMBEDDING = self.embeddings(torch.tensor([49407]))[0, 0, :]
    
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
                
    def to_embedding(self, input_ids, token_summary_idx=None):
        word_embeddings = self.embeddings(input_ids)
        summarized_word_embeddings = self.sum_batch_entity_embeddings(word_embeddings, token_summary_idx) \
            if token_summary_idx is not None else word_embeddings
        return summarized_word_embeddings
    
    def visual_forward(self, pixel_values):

        self.visual_embeddings = self.vision_encoder(pixel_values).last_hidden_state

        # Map image patches to text embeddings
        return self.vision_projector(self.visual_embeddings)

    def lang_forward(self, img_src, txt_tgt):
        # Send image seq and text seq to lang model
        return self.label_head(self.decoder(
            img_src,
            txt_tgt,
            tgt_mask=self.decoder.generate_square_subsequent_mask(
                txt_tgt.shape[1],
                device=txt_tgt.device,
            )
        ))

    def forward(self, pixel_values, text_values, token_summary_idx=None):    

        # Get visual features
        visual_features = self.visual_forward(pixel_values)

        # Lang Model
        return self.lang_forward(visual_features, text_values)

class AutoSegDecoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.decoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=768,
                nhead=8,
                dim_feedforward=2048,
                dropout=0.1,
                activation=nn.GELU(),
                batch_first=True,
            ),
            num_layers=6,
        )

        self.mask_head = nn.ConvTranspose2d(
            in_channels=768,
            out_channels=1,
            kernel_size=7,
            stride=7,
        )

    def forward(self, image_features):
        features = self.decoder(image_features)[:,1:,:]
        features = features.view(features.shape[0], features.shape[-1], 7, 7)
        output = self.mask_head(features)
        return F.interpolate(output, size=(224, 224), mode='bilinear')


class AutoSeg(nn.Module):

    def __init__(self):
        super().__init__()

        # To Implement
        # FiLM conditional nomenclature
        # Short cut connection ?
        # Transformer Encoder ?
        # Transformer Decoder for next token/mask prediction
        # Decoder for object classification and mask prediction

        # Note that the model is supposed to find the first prediction automatically
        # which may be formulated as an image classification task?
        # Investigate how CLIP actually work

        # Assuming CLIP is using causal langauge mask
        # The architecute should combine features from 2 encoders respectively
        # The decoder should a transformer decoder
        # 2 Prediction heads predict a class label and a mask at the same time
        # Combine language modeling loss and segmentation loss

        # Linear Mapping to apply to every image token
        # self.image_feature_projection = nn.Sequential(
        #     nn.Linear(768, 512)
        # )# MLP may be preferable

        self.encoders = CLIPLang()

        self.film_mul = nn.Linear(512, 768)
        self.film_add = nn.Linear(512, 768)
        
        self.autoseg_decoder = AutoSegDecoder()

        # Prediction Head Check MaskFormer architecture
        # It seems MLP should suffice for both ends
        # Anticipated working case:
        # <bos> A photo of {label}
        # <bos> A photo of cat {label}
        # <bos> A photo of cat sofa {label}
        # <bos> A photo of cat sofa <eos>

    def visual_forward(self, pixel_values):
        self.encoders(pixel_values)
        return self.encoders.visual_embeddings

    def forward(self, pixel_values, text_values, **kargs):        
        
        # Get text embeddings
        lang_output = self.encoders(pixel_values, text_values)

        # Get image embeddings
        img_output = self.encoders.visual_embeddings

        mask_outputs = []
        for batch_embeddings in lang_output.permute(1, 0, 2):
            conditioned_img = self.film_mul(batch_embeddings).unsqueeze(1) * img_output +\
                 self.film_add(batch_embeddings).unsqueeze(1)
                
            batch_masks = self.autoseg_decoder(conditioned_img)
            mask_outputs.append(batch_masks)
        mask_outputs = torch.cat(mask_outputs, dim=1)



        return mask_outputs, lang_output
        

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
        




