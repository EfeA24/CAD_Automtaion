import torch.nn as nn
import torch
from transformers import BertTokenizer, BertModel


MODEL_NAME_DICT={"bert_large_uncased":"google-bert/bert-large-uncased"}

def prepare_cross_attention_mask_batch(mask, cad_seq_len=271):
    if mask.shape[0] > 1:
        length=mask.shape[1]
        batch_size=mask.shape[0]
        mask = mask.reshape(batch_size, 1, length)
    mask = torch.tile(mask, (1, cad_seq_len, 1))  # (512) -> (271, 512)
    mask = torch.where(
        mask, -torch.inf, 0
    )  # Changing the [True,False] format to [0,-inf] format

    return mask

class TextEmbedder(nn.Module):
    def __init__(self, model_name:str, cache_dir:str, max_seq_len:int):
        super(TextEmbedder, self).__init__()
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.max_seq_len = max_seq_len
        self.model_name = MODEL_NAME_DICT.get(model_name, "bert_large_uncased")
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name, cache_dir=cache_dir)
        self.model = BertModel.from_pretrained(
                self.model_name, cache_dir=cache_dir, max_position_embeddings=max_seq_len
            ).to(device)
    
    def get_embedding(self, texts:list[str]):
        if isinstance(texts, str):
            texts = [texts]
        with torch.no_grad():
                input_ids = self.tokenizer(
                    texts,
                    return_tensors="pt",
                    max_length=self.max_seq_len,
                    truncation=True,
                    padding=True,
                ).to("cuda")
                all_output = self.model(**input_ids)

                embedding = all_output[0]
                key_padding_mask = (
                    (input_ids["attention_mask"] == 0)
                )
                
        return embedding, key_padding_mask

    @staticmethod
    def from_config(config: dict):
        return TextEmbedder(
            **config
        )
        
