
import torch
import torch.nn as nn
from torchmetrics.image import StructuralSimilarityIndexMeasure
from config import DEVICE, ALPHA


class MSELoss(nn.Module):

    def __init__(self):
        super().__init__()

        self.mse = nn.MSELoss()

    def forward(self, prediction, target):

        return self.mse(prediction, target)
    



class HybridLoss(nn.Module):

    def __init__(self, alpha=ALPHA):
        super().__init__()

        self.alpha = alpha

        self.mse = nn.MSELoss()

        self.ssim = StructuralSimilarityIndexMeasure(
            data_range=1.0
        ).to(DEVICE)

    def forward(self, prediction, target):

        mse_loss = self.mse(
            prediction,
            target
        )

        ssim_value = self.ssim(
            prediction,
            target
        )

        hybrid_loss = (
            self.alpha * mse_loss
            +
            (1 - self.alpha) * (1 - ssim_value)
        )

        return hybrid_loss
    

    