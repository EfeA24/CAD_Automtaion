import os, sys

sys.path.append("/".join(os.path.abspath(__file__).split("/")[:-3]))

import torch.nn as nn
import torch
from Cad_VLM.models.layers.adaptive_layer import TextAdaptiveLayer
from Cad_VLM.models.layers.text_embed import TextEmbedder, prepare_cross_attention_mask_batch
from Cad_VLM.models.decoder import CADDecoder
from Cad_VLM.models.utils import count_parameters



class Text2CAD(nn.Module):
    """
    Text2CAD: Generating CAD Designs from beginner-to-expert text prompts

    """

    def __init__(self, text_config, cad_config):
        super().__init__()
        
        # Base Text Embedder
        self.base_text_embedder = TextEmbedder.from_config(text_config['text_embedder'])
        # Adaptive Layer to fine tune text embeddings
        self.adaptive_layer = TextAdaptiveLayer.from_config(text_config['adaptive_layer'])

        # Transformer Decoder for CAD sequence generation
        self.cad_decoder = CADDecoder.from_config(cad_config)
        
        self.cad_seq_len= cad_config["cad_seq_len"]-1

        self.attention_scores = dict()

    def test_decode(self, 
                        texts: list[str],
                        maxlen:int,
                        nucleus_prob,
                        topk_index,
                        device='cuda' if torch .cuda.is_available() else 'cpu'
                        ):
        """
        Auto-regressively decode CAD sequence from text prompts
        Args:
        - texts: list of text prompts
        - maxlen: maximum length of the generated CAD sequence
        - nucleus_prob: nucleus sampling probability
        - topk_index: top-k sampling index
        - device: device to run the model
        """
        
        ZE, key_padding_mask = self.base_text_embedder.get_embedding(texts)
        ca_mask={"attn_mask": prepare_cross_attention_mask_batch(key_padding_mask, cad_seq_len=1), 
                 "key_padding_mask": key_padding_mask}
        ZE, _  = self.adaptive_layer(ZE,
            {
                "attn_mask": None,
                "key_padding_mask": ca_mask["key_padding_mask"],
            },
            False,)
        S_output = self.cad_decoder.decode(
                        ZE=ZE, 
                        cross_attn_mask_dict=ca_mask, 
                        maxlen=maxlen, 
                        nucleus_prob=nucleus_prob,
                        topk_index=topk_index, 
                        device=device)
        
        return S_output

    def forward(
        self,
        vec_dict: dict,
        texts: list[str],
        mask_cad_dict: dict,
        metadata: bool = False,
    ):
        """
        vec_dict: dict contains cad_vec, flag_vec, and index_vec
        texts: list of text prompts
        mask_cad_dict: dict contains attention mask and key_padding_mask for CAD sequence
        metadata: bool to return attention scores
        """
        # ------------ Get the initial text embeddings ------------ #
        
        T, key_padding_mask = self.base_text_embedder.get_embedding(texts)
        ca_mask={"attn_mask": prepare_cross_attention_mask_batch(key_padding_mask, cad_seq_len=self.cad_seq_len), 
                 "key_padding_mask": key_padding_mask}
        
        # ------------ Pass the text embedding through the adaptive layer ------------ #
        T, text_attn_scores = self.adaptive_layer(
            T,
            {
                "attn_mask": None,
                "key_padding_mask": ca_mask["key_padding_mask"],
            },
            metadata,
        )
        if text_attn_scores is not None:
            self.attention_scores.update(text_attn_scores)

        # ------------ Pass the text embedding through the CAD Decoder as context ------------ #
        S_output, cad_attn_scores = self.cad_decoder(
            vec_dict, T, mask_cad_dict, ca_mask, metadata
        )
        if cad_attn_scores is not None:
            self.attention_scores.update(cad_attn_scores)

        if metadata:
            return S_output, self.attention_scores
        else:
            return S_output, None

    def total_parameters(self, description=False, in_millions=False):
        num_params = count_parameters(self, description)
        if in_millions:
            num_params_million = num_params / 1_000_000  # Convert to millions
            print(f"Number of Parameters: {num_params_million:.1f}M")
        else:
            num_params = count_parameters(self, description)
            print(f"Number of Parameters: {num_params}")
        
    def get_trainable_state_dict(self):
        # Get the state dict of the model which are trainable parameters
        return {
            k: v for k, v in self.state_dict().items() if "base_text_embedder" not in k.split(".")
        }