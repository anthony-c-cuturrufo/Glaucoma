from monai.networks.blocks.patchembedding import PatchEmbeddingBlock
from monai.networks.blocks.transformerblock import TransformerBlock
from monai.networks.nets.vit import ViT
import torch
import torch.nn as nn
from collections.abc import Sequence


class DualViT(ViT):
    def __init__(
        self,
        in_channels_1: int,
        img_size_1: Sequence[int] | int,
        patch_size_1: Sequence[int] | int,
        in_channels_2: int,
        img_size_2: Sequence[int] | int,
        patch_size_2: Sequence[int] | int,
        hidden_size: int,
        num_heads: int,
        dropout_rate: float,
        proj_type: str = 'conv',
        pos_embed_type: str = 'learnable',
        spatial_dims: int = 3,   # Set default value for num_heads
        *args,
        **kwargs
    ):
        # Set hidden_size and num_heads before calling super().__init__
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.proj_type = proj_type
        self.pos_embed_type = pos_embed_type
        self.dropout_rate = dropout_rate
        self.spatial_dims = spatial_dims
        
        # Now call the parent class's init
        super().__init__(
            in_channels=in_channels_1, 
            img_size=img_size_1, 
            patch_size=patch_size_1,
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            proj_type=self.proj_type,
            pos_embed_type=self.pos_embed_type,
            dropout_rate=self.dropout_rate,
            spatial_dims=self.spatial_dims,
            *args, 
            **kwargs
        )

        # Rest of your initialization code...


        # Now it's safe to use self.hidden_size and self.num_heads
        self.patch_embedding_2 = PatchEmbeddingBlock(
            in_channels=in_channels_2,
            img_size=img_size_2,
            patch_size=patch_size_2,
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            proj_type=self.proj_type,
            pos_embed_type=self.pos_embed_type,
            dropout_rate=self.dropout_rate,
            spatial_dims=self.spatial_dims,
        )


    def forward(self, x1, x2):
        # Embed both scans
        x1 = self.patch_embedding(x1)
        x2 = self.patch_embedding_2(x2)

        # Concatenate patches from both scans
        x = torch.cat((x1, x2), dim=1)

        # Add class token if classification is enabled
        if hasattr(self, "cls_token"):
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)

        # Process through transformer blocks
        hidden_states_out = []
        for blk in self.blocks:
            x = blk(x)
            hidden_states_out.append(x)

        # Apply final layers
        x = self.norm(x)
        if hasattr(self, "classification_head"):
            x = self.classification_head(x[:, 0])

        return x
