import torch
import torch.nn as nn


class CELoss(nn.Module):
    """
    Cross Entropy Loss for Text2CAD
    """

    def __init__(self, device):
        super(CELoss, self).__init__()

        self.ce_cad = nn.CrossEntropyLoss(reduction="none", label_smoothing=0.1)
        self.ce_pc = nn.CrossEntropyLoss()
        self.mseloss = nn.MSELoss()

    def forward(self, cad_dict: dict):
        """
        cad_dict: dictionary containing 'pred', 'target' and 'key_padding_mask' key.
                pred: shape (B,N,2)
                target: shape (B,N)
                key_padding_mask: shape (B,N)
        """

        key_padding_mask = cad_dict["key_padding_mask"]
        loss = []
        if cad_dict["key_padding_mask"] is not None:
            self.loss_seq_x = torch.sum(
                self.ce_cad(
                    cad_dict["pred"][:, :, 0].permute(0, 2, 1),
                    cad_dict["target"][:, :, 0].long(),
                )
                * key_padding_mask[:, :, 0]
            ) / torch.sum(key_padding_mask[:, :, 0] * 1)
            self.loss_seq_y = torch.sum(
                self.ce_cad(
                    cad_dict["pred"][:, :, 1].permute(0, 2, 1),
                    cad_dict["target"][:, :, 1].long(),
                )
                * key_padding_mask[:, :, 1]
            ) / torch.sum(key_padding_mask[:, :, 1] * 1)

        self.loss_seq = (self.loss_seq_x + self.loss_seq_y) / 2
        loss_keys = ["loss_seq"]

        result_dict = {key: getattr(self, key).detach().item() for key in loss_keys}

        loss = self.loss_seq
        return loss, result_dict
