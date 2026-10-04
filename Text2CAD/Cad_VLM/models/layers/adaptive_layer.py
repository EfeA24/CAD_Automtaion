import torch
import copy
import torch.nn as nn
import os, sys

sys.path.append("..")
sys.path.append("/".join(os.path.abspath(__file__).split("/")[:-3]))


from Cad_VLM.models.layers.attention import MultiHeadAttention
from Cad_VLM.models.layers.functional import FeedForwardLayer
from rich import print
from Cad_VLM.models.layers.utils_decode import generate_attention_mask
from Cad_VLM.models.utils import count_parameters
from typing import Optional


class TextAdaptiveLayer(nn.Module):
    """
    Adaptive Layer for Text Embeddings

    """

    def __init__(self, in_dim: int, out_dim: int, num_heads: int, dropout: float):
        super(TextAdaptiveLayer, self).__init__()

        # Multi-Head Self Attention for CAD Sequence
        # TODO: Check if Flash Attention is implemented
        self.sa_seq = MultiHeadAttention(
            input_dim=in_dim,
            embed_dim=in_dim,
            dropout=dropout,
            num_heads=num_heads,
        )

        # LayerNormalization
        self.norm_seq = nn.ModuleDict(
            {"norm_1": nn.LayerNorm(in_dim), "norm_2": nn.LayerNorm(in_dim)}
        )

        # Dropout
        self.dp_seq = nn.ModuleDict(
            {"dropout_1": nn.Dropout(dropout), "dropout_2": nn.Dropout(dropout)}
        )

        # Feed forward Networks
        self.ffl_seq = FeedForwardLayer(input_dim=in_dim)

        if in_dim != out_dim:
            # Downsampler
            self.downsampler = nn.Linear(in_dim, out_dim)

        # Attention Scores
        self.attention_scores = dict()

    def forward(
        self,
        T: Optional[torch.Tensor],
        mask_prompt_dict: dict,
        metadata: bool = False,
    ):
        """
        T: tensor of shape (bs, num_seq, emb_dim). Text Embedding
        mask_prompt_dict: dictionary with keys "attn_mask", "key_padding_mask"
        metadata: boolean. To save attention weights
        """

        # self_attn_mask_dict = mask_cad_dict.copy()

        # ? <----------  TEXT EMBEDDING SELF-ATTENTION  ---------->
        T2 = self.norm_seq["norm_1"](T)  # (bs,num_seq,emb_dim)
        # exit()
        T2, T_score = self.sa_seq(
            T2,
            T2,
            T2,
            key_padding_mask=mask_prompt_dict["key_padding_mask"],
            attn_mask=mask_prompt_dict["attn_mask"],
        )  # (bs,num_seq,emb_dim) (Self-Attention)

        # (bs,num_seq,emb_dim) (Dropout + Addition)
        T = T + self.dp_seq["dropout_1"](T2)
        T2 = self.norm_seq["norm_2"](T)  # (bs,num_seq,emb_dim) (Normalization)

        # ? <---------- FEED-FORWARD + DROPOUT + ADDITION + DOWN-SAMPLER    ---------->
        T = T + self.dp_seq["dropout_2"](self.ffl_seq(T2))

        if hasattr(self, "downsampler"):
            T = self.downsampler(T)

        if metadata:
            # Add the cross attention scores (metadata)
            self.attention_scores["text_sattn"] = T_score
            return T, self.attention_scores
        else:
            return T, None

    def total_parameters(self, description=False, in_millions=False):
        num_params = count_parameters(self, description)
        if in_millions:
            num_params_million = num_params / 1_000_000  # Convert to millions
            print(f"Number of Parameters: {num_params_million:.1f}M")
        else:
            num_params = count_parameters(self, description)
            print(f"Number of Parameters: {num_params}")

    @staticmethod
    def from_config(config):
        return TextAdaptiveLayer(**config)


if __name__ == "__main__":
    adaptive_layer = TextAdaptiveLayer(4096, 4096, 8, 0.1).cuda()
    input_tensor = torch.rand(32, 512, 4096).cuda()

    attn_mask = generate_attention_mask(512)

    output, attn_weight = adaptive_layer(
        input_tensor,
        {
            "attn_mask": None,
            "key_padding_mask": torch.randint(0, 2, (32, 512)).bool().cuda(),
        },
        metadata=True,
    )
    print(output.shape)

    print(attn_weight)
    print(adaptive_layer.total_parameters())
