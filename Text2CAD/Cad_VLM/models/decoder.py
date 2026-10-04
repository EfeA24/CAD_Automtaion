import torch
import copy
import torch.nn as nn
import os, sys

sys.path.append("/".join(os.path.abspath(__file__).split("/")[:-3]))
from CadSeqProc.utility.macro import *
from CadSeqProc.utility.utils import (
    generate_attention_mask,
    create_flag_vec,
    create_index_vec,
    top_p_sampling,
)
from Cad_VLM.models.layers.embedder import CADSequenceEmbedder, PositionalEncodingSinCos
from Cad_VLM.models.layers.attention import CrossAttention, MultiHeadAttention
from Cad_VLM.models.layers.functional import FeedForwardLayer
from Cad_VLM.models.utils import count_parameters
from rich import print
from typing import Optional


class CADDecoder(nn.Module):
    """
    Decodes the discrete latent codes into CAD Sequence autoregressively.

    Args:
        cad_class_info (dict, required): A dictionary containing information about CAD classes. Check `DataProc.CadSeqProc.utility.macro`
        tdim (int, required): Dimensionality of the continuous latent space.
        cdim (int, required): Dimensionality of the cad embedding.
        num_layers (int, required): Number of transformer layers.
        num_heads (int, required): Number of attention heads in the transformer.
        dropout (float, required): Dropout probability.
        mode (str, required): Either "train" or "test" for specifying the mode of operation.
        ca_level_start (int, required): The starting level of class attention layers.
        device (str, required): Device to run the model on, e.g., "cuda" or "cpu".

    Example:
       To initialize the CADDecoder, you can use the following example:

        ```python
        decoder = CADDecoder(
            cad_class_info={"one_hot_size": 267, "index_size": 11, "flag_size": 12}, tdim=128, cdim=128, num_layers=8,
            num_heads=8, dropout=0.1, mode="train", ca_level_start=0, device="cuda",
        )
        ```

    """

    def __init__(
        self,
        cad_class_info: dict,
        tdim: int,
        cdim: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        ca_level_start: int,
        device: str,
    ):
        super(CADDecoder, self).__init__()

        self.num_layers = num_layers

        # For Initial Sequence Embedding
        self.seq_embed = CADSequenceEmbedder(
            one_hot_size=cad_class_info["one_hot_size"],
            flag_size=cad_class_info["flag_size"],
            index_size=cad_class_info["index_size"],
            d_model=cdim,
            device=device,
        )

        # Positional Encoding
        self.pe = PositionalEncodingSinCos(
            embedding_size=cdim, max_seq_len=MAX_CAD_SEQUENCE_LENGTH, device=device
        )

        # List of booleans indicating which decoder layers support cross-attention
        self.use_ca = [False] * ca_level_start + [True] * (num_layers - ca_level_start)

        # A stack of num_layers x Decoder Layer
        self.cad_decoder_layers = nn.ModuleList(
            [
                copy.deepcopy(
                    CADDecoderLayer(
                        tdim=tdim,
                        cdim=cdim,
                        num_heads=num_heads,
                        block_level=i,
                        dropout=dropout,
                        use_ca=self.use_ca[i],
                        device=device,
                    )
                )
                for i in range(self.num_layers)
            ]
        )

        # Out Layer for Sequence Prediction
        self.seq_output_x = nn.Sequential(
            nn.Linear(in_features=cdim, out_features=cad_class_info["one_hot_size"])
        )
        self.seq_output_y = nn.Sequential(
            nn.Linear(in_features=cdim, out_features=cad_class_info["one_hot_size"])
        )

        # Metadata
        self.attention_scores = dict()

    def forward(
        self,
        vec_dict: dict,
        ZE: Optional[torch.tensor],
        mask_cad_dict: dict,
        cross_attn_mask_dict: dict = {"attn_mask": None, "key_padding_mask": None},
        metadata: bool = False,
    ):
        """
        vec_dict: dictionary with keys "cad_vec", "flag_vec", "index_vec"
        ZE: tensor of shape (batch, num_code, emd_dim). Context
        mask_cad_dict: dictionary with keys "key_padding_mask" and "attn_mask". SELF ATTENTION MASK
        cross_attn_mask_dict: dictionary with keys "attn_mask" and "key_padding_mask". CROSS ATTENTION MASK
        metadata: boolean indicating whether attention weights are saved. Turn off during training
        """
        num_seq = vec_dict["cad_vec"].shape[1]
        # Token Embedding and positional encoding
        S = self.pe(num_seq) + self.seq_embed(
            vec_dict, mask_cad_dict["key_padding_mask"]
        )  # (B,N1,cdim)
        # Pass through Decoder Layers
        for i in range(self.num_layers):
            S, self.attention_scores[f"block_level_{i}"] = self.cad_decoder_layers[i](
                S,
                ZE=ZE,
                mask_cad_dict=mask_cad_dict,
                cross_attn_mask_dict=cross_attn_mask_dict,
                metadata=metadata,
            )
        Sx = self.seq_output_x(S).unsqueeze(dim=2)  # (B,1,N1,one_hot_size)
        Sy = self.seq_output_y(S).unsqueeze(dim=2)  # (B,N1,1,one_hot_size)

        S = torch.cat([Sx, Sy], dim=2)  # (B,N1,2*one_hot_size

        # Save the metadata
        if metadata:
            self.metadata = {"attention_scores": self.attention_scores}
            return S, self.metadata
        else:
            return S, None

    @staticmethod
    def from_config(config):
        """Initialize the CADDecoder class from a config file"""
        cad_decoder = CADDecoder(
            cad_class_info=CAD_CLASS_INFO,
            tdim=config["tdim"],
            cdim=config["cdim"],
            num_layers=config["num_layers"],
            num_heads=config["num_heads"],
            dropout=config["dropout"],
            ca_level_start=config["ca_level_start"],
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        return cad_decoder

    def decode(
        self,
        ZE,
        cross_attn_mask_dict,
        maxlen,
        nucleus_prob,
        topk_index,
        device,
    ):
        """
        Autoregressive Prediction for test and validation.
        :param new_cad_seq_dict: dictionary containing keys "cad_vec", "flag_vec", "index_vec". cad_vec shape (batch, 1, 2) and flag and index vec (batch,1,1)
        :param ZE: tensor of shape (batch, num_latent, emb_dim)
        :param cross_attn_mask_dict: dictionary containing keys "attn_mask", "key_padding_mask". Keep both None.
        :maxlen maximum length of sequence.
        :nucleus_prob: Probability for Nucleus sampling. If 0, then top1-sampling.
                        More probability for more diversity but less valid CAD models and less probability for less diversity but more correct CAD models.
        :device: "cuda" or "cpu"
        """
        self.eval()
        num_texts = ZE.shape[0]
        new_cad_seq_dict={
            "cad_vec": torch.tensor([[[1, 0]]]).repeat(num_texts, 1, 1).to(device),
            "flag_vec": torch.zeros(num_texts, 1).int().to(device),
            "index_vec": torch.zeros(num_texts, 1).int().to(device),
        }
        
        # NOTE: Iteratively run the forward method till the end token is predicted.
        for t in range(1, maxlen):
            cad_mask = {
                "attn_mask": cross_attn_mask_dict["attn_mask"].repeat(1, t, 1),
                "key_padding_mask": cross_attn_mask_dict["key_padding_mask"],
            }
            cad_pred, _ = self(
                new_cad_seq_dict,
                ZE,
                {
                    "attn_mask": generate_attention_mask(t, t, device=device),
                    "key_padding_mask": (new_cad_seq_dict["cad_vec"] == 0),
                },
                cad_mask,
                False,
            )

            # --------------------------------- Sampling --------------------------------- #
            # Hybrid-Sampling
            if nucleus_prob == 0:
                if t == 1:  # NOTE: Remove this part for top-1 sampling
                    new_token = torch.topk(cad_pred, topk_index, dim=-1).indices[
                        :, t - 1 : t, :, -1
                    ]
                else:
                    # NOTE: Keep this part only for top-1 sampling
                    new_token = torch.argmax(cad_pred, dim=-1)[:, t - 1 : t]
            # Nucleus Sampling
            else:
                new_token = torch.cat(
                    [
                        top_p_sampling(cad_pred[:, t - 1 : t, 0], nucleus_prob),
                        top_p_sampling(cad_pred[:, t - 1 : t, 1], nucleus_prob),
                    ],
                    axis=-1,
                )

            # ------------------------------ CAD Sequence Update ------------------------------ #
            # Add the new token (no masking here)
            new_cad_seq_dict["cad_vec"] = torch.cat(
                [new_cad_seq_dict["cad_vec"], new_token], axis=1
            )

            # ------------------------------ Flag generation ----------------------------- #
            # Create flag seq (Very important. Wrong flag may result in invalid model)
            new_cad_seq_dict["flag_vec"] = torch.cat(
                [
                    new_cad_seq_dict["flag_vec"],
                    create_flag_vec(
                        new_cad_seq_dict["cad_vec"], new_cad_seq_dict["flag_vec"]
                    ),
                ],
                axis=1,
            )

            # ----------------------------- Index Generation ----------------------------- #
            # Create index seq  (Very important. Wrong index may result in invalid model)
            new_cad_seq_dict["index_vec"] = torch.cat(
                [
                    new_cad_seq_dict["index_vec"],
                    create_index_vec(
                        new_cad_seq_dict["cad_vec"], new_cad_seq_dict["index_vec"]
                    ),
                ],
                axis=1,
            )

            # ------------------------- Masking the dummy tokens ------------------------- #
            # Mask the dummy tokens in the new CAD tokens (Very important. Wrong masking may result in inaccurate model)

            end_tokens=torch.logical_or(new_cad_seq_dict['cad_vec'][:,:,0] <= END_TOKEN.index("END_EXTRUSION"),new_cad_seq_dict['flag_vec']>0)
    

            num_tokens=new_cad_seq_dict["cad_vec"][
                end_tokens
            ].shape[0]

            mask = torch.cat(
                [
                    torch.ones((num_tokens, 1), dtype=torch.int32),
                    torch.zeros((num_tokens, 1), dtype=torch.int32),
                ],
                axis=1,
            ).to(device)
            
            new_cad_seq_dict["cad_vec"][
                end_tokens
            ] *= mask

        return new_cad_seq_dict

    def total_parameters(self, description=False, in_millions=False):
        num_params = count_parameters(self, description)
        if in_millions:
            num_params_million = num_params / 1_000_000  # Convert to millions
            print(f"Number of Parameters: {num_params_million:.1f}M")
        else:
            num_params = count_parameters(self, description)
            print(f"Number of Parameters: {num_params}")


class CADDecoderLayer(nn.Module):
    """
    CAD Decoder Layers

    """

    def __init__(
        self,
        tdim=128,
        cdim=128,
        num_heads=8,
        block_level=0,
        dropout=0.1,
        use_ca=True,
        device="cuda",
    ):
        super(CADDecoderLayer, self).__init__()

        # Multi-Head Self Attention for CAD Sequence
        # TODO: Check if Flash Attention is implemented
        #! Note: Dropout is set to 0 for attention otherwise the sum of attention weights > 1
        self.sa_seq = MultiHeadAttention(
            input_dim=cdim,
            embed_dim=cdim,
            dropout=0,
            num_heads=num_heads,
        )
        self.use_ca = use_ca

        if use_ca:
            # Cross Attention between discrete latent vectors and cad features
            # TODO: Implement Flash attention here
            self.ca_ze_seq = CrossAttention(
                input_dim_list=[cdim, tdim],
                output_dim=cdim,
                query_name="cad",
                context_1_name="vq",
                dropout=0,
                block_level=block_level,
            )

        # Text Embedding Downsampler
        self.downsampler = nn.Linear(tdim, cdim)

        # LayerNormalization
        self.norm_seq = nn.ModuleDict(
            {
                "norm_1": nn.LayerNorm(cdim),
                "norm_2": nn.LayerNorm(cdim),
                "norm_3": nn.LayerNorm(cdim),
            }
        )

        # Dropout
        self.dp_seq = nn.ModuleDict(
            {
                "dropout_1": nn.Dropout(dropout),
                "dropout_2": nn.Dropout(dropout),
                "dropout_3": nn.Dropout(dropout),
            }
        )

        # Feed forward Networks
        self.ffl_seq = FeedForwardLayer(input_dim=cdim)

        # Attention Scores
        self.attention_scores = dict()

    def forward(
        self,
        S: Optional[torch.Tensor],
        ZE: Optional[torch.Tensor],
        mask_cad_dict: dict,
        cross_attn_mask_dict: dict = {"attn_mask": None, "key_padding_mask": None},
        metadata: bool = False,
    ):
        """
        S: tensor of shape (bs, num_seq, emb_dim)
        ZE: tensor of shape (bs, num_code, emd_dim)
        mask_cad_dict: dictionary with keys "attn_mask", "key_padding_mask"
        cross_attn_mask_dict: dictionary with keys "attn_mask", "key_padding_mask"
        metadata: boolean. To save attention weights
        """

        self_attn_mask_dict = mask_cad_dict.copy()
        self_attn_mask_dict["key_padding_mask"] = torch.all(
            mask_cad_dict["key_padding_mask"], axis=2
        )

        # ? <----------  CAD SEQUENCE SELF-ATTENTION  ---------->
        S2 = self.norm_seq["norm_1"](S)  # (bs,num_seq,emb_dim)
        # exit()
        S2, S_score = self.sa_seq(
            S2,
            S2,
            S2,
            key_padding_mask=self_attn_mask_dict["key_padding_mask"],
            attn_mask=self_attn_mask_dict["attn_mask"],
        )  # (bs,num_seq,emb_dim) (Self-Attention)

        # (bs,num_seq,emb_dim) (Dropout + Addition)
        S = S + self.dp_seq["dropout_1"](S2)
        S2 = self.norm_seq["norm_2"](S)  # (bs,num_seq,emb_dim) (Normalization)

        # ? <----------  CROSS-ATTENTION BETWEEN LATENT EMBEDDING AND CAD EMBEDDING  ---------->
        if self.use_ca:
            if hasattr(self, "downsampler"):
                ZE = self.downsampler(ZE)
            S3, ZE_S_score = self.ca_ze_seq(
                S2,
                ZE,
                ZE,
                key_padding_mask=cross_attn_mask_dict["key_padding_mask"],
                attn_mask=cross_attn_mask_dict["attn_mask"],
            )

            # ? <----------  (CROSS-ATTENDED FEATURES + SELF-ATTENDED FEATURES) + DROPOUT + NORMALIZATION LAYER  ---------->
            S = S + self.dp_seq["dropout_2"](S3)
            S2 = self.norm_seq["norm_3"](S)  # (bs,num_seq,emb_dim)
            self.attention_scores["ca"] = ZE_S_score

        # ? <---------- FEED-FORWARD + DROPOUT + ADDITION +    ---------->
        S = S + self.dp_seq["dropout_3"](self.ffl_seq(S2))

        # Add the cross attention scores (metadata)
        self.attention_scores["sa"] = S_score

        return S, self.attention_scores


if __name__ == "__main__":
    pass
    