import torch
from torch import nn
from transformers import CLIPProcessor, CLIPModel

class CoSeg(nn.module):

    def __init__(self):
        clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.vision_encoder = clip.vision_model
        self.text_model = clip.text_model
        
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
        self.image_feature_projection = nn.Linear(768, 512) # MLP may be preferable

        self.decoder = nn.Transformer(d_model=512, nhead=8, batch_first=True)

        # Prediction Head Check MaskFormer architecture
        # It seems MLP should suffice for both ends
        # Anticipated working case:
        # <bos> A photo of {label}
        # <bos> A photo of cat {label}
        # <bos> A photo of cat sofa {label}
        # <bos> A photo of cat sofa <eos>

        self.label_head = nn.MLP(
            512,
            [1024, 2048, 5096, self.text_model.embedding.num_embeddings]
        )

        self.pixel_head = nn.MLP(
            512,
            [1024, 2048, 5096, 224 * 224]
        )
        
    
    def forward(self, pixel_values, input_ids):
        pass
