import torch
from torch import nn
from transformers import CLIPProcessor, CLIPModel
import numpy as np


class AutoSegDecoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=512, nhead=8, batch_first=True),
            num_layers=6,
        )

        self.label_head = nn.Sequential(
            nn.Linear(512, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 2048),
            nn.LeakyReLU(),
            nn.Linear(2048, 5096),
            nn.LeakyReLU(),
            nn.Linear(5096, 512),
        )

        self.pixel_head = nn.Sequential(
            nn.Linear(512, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 2048),
            nn.LeakyReLU(),
            nn.Linear(2048, 5096),
            nn.LeakyReLU(),
            nn.Linear(5096, 224 * 224),
        )

    def create_causal_mask(self, size):
        return torch.triu(torch.full((size, size), float("-inf")), diagonal=1).to(0)

    def sum_batch_entity_embeddings(self, batch_embeddings, batch_labels):
        output_seq_length = torch.max(batch_labels + 1)

        def sum_entity_embeddings_torch(embeddings, labels):
            unique_labels, inverse_indices = torch.unique_consecutive(
                labels, return_inverse=True
            )
            summed_embeddings = torch.zeros(
                (output_seq_length, embeddings.size(1)), device=embeddings.device
            )
            for i, idx in enumerate(inverse_indices.unique()):
                summed_embeddings[i] = embeddings[inverse_indices == idx].sum(dim=0)
            return summed_embeddings

        batch_summed_embeddings = []
        for embeddings, labels in zip(batch_embeddings, batch_labels):
            summed_embeddings = sum_entity_embeddings_torch(embeddings, labels)
            batch_summed_embeddings.append(summed_embeddings)

        return torch.stack(batch_summed_embeddings)

    def forward(self, vk_seq, q_seq, token_summary_idx=None):

        # Query: text tokens
        # Key & Value: image tokens
        transformer_output = self.decoder(
            q_seq,
            vk_seq,
            tgt_mask=self.create_causal_mask(q_seq.shape[1]),
        )

        label_logit = self.label_head(transformer_output)

        # Pool tokens correspond to the same entity for mask generation
        if token_summary_idx is not None:
            pooled_hidden_state = self.sum_batch_entity_embeddings(
                transformer_output, token_summary_idx
            )
        else:
            pooled_hidden_state = transformer_output

        masks = self.pixel_head(pooled_hidden_state).reshape(
            pooled_hidden_state.shape[:-1] + (224, 224)
        )

        return masks, label_logit


class AutoSeg(nn.Module):

    def __init__(self, freeze_clip=True):
        super().__init__()
        clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.vision_encoder = clip.vision_model
        self.text_encoder = clip.text_model

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

        self.image_feature_projection = clip.visual_projection
        self.text_feature_projection = clip.text_projection

        for module in [
            self.vision_encoder,
            self.text_encoder,
            self.image_feature_projection,
            self.text_feature_projection,
        ]:
            for param in module.parameters():
                param.requires_grad = False

        self.autoseg_decoder = AutoSegDecoder()

        # Prediction Head Check MaskFormer architecture
        # It seems MLP should suffice for both ends
        # Anticipated working case:
        # <bos> A photo of {label}
        # <bos> A photo of cat {label}
        # <bos> A photo of cat sofa {label}
        # <bos> A photo of cat sofa <eos>

    def project_to_clip_space(self, input_ids):
        text_hidden_state = self.text_encoder(input_ids=input_ids).last_hidden_state
        projected_text_hidden_state = self.text_feature_projection(text_hidden_state)
        return projected_text_hidden_state

    def forward(
        self, pixel_values, input_ids, new_image=True, token_summary_idx=None, **kargs
    ):

        # Process image
        if new_image or self.image_conditional is None:
            image_hidden_state = self.vision_encoder(
                pixel_values=pixel_values
            ).last_hidden_state
            # Save the projected image hidden states for future prediction
            self.image_conditional = self.image_feature_projection(image_hidden_state)

        # Process text
        projected_text_hidden_state = self.project_to_clip_space(input_ids)

        # Decode
        masks, label_logit = self.autoseg_decoder(
            self.image_conditional, projected_text_hidden_state, token_summary_idx
        )

        return masks, label_logit


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
    labels = torch.arange(N * L).to(similarity.device)

    # Compute the loss
    loss = nn.functional.cross_entropy(similarity, labels)

    return loss
