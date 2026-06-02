import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.utils.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class FFTLoss(nn.Module):

    def __init__(self, loss_weight: float = 1.0, reduction: str = 'mean') -> None:
        super().__init__()
        self.loss_weight = loss_weight
        self.reduction   = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

        pred_fft   = torch.fft.rfft2(pred,   norm='ortho')
        target_fft = torch.fft.rfft2(target, norm='ortho')

        pred_mag   = torch.abs(pred_fft)
        target_mag = torch.abs(target_fft)

        pred_log   = torch.log(pred_mag   + 1e-8)
        target_log = torch.log(target_mag + 1e-8)

        loss = F.l1_loss(pred_log, target_log, reduction=self.reduction)

        return loss * self.loss_weight