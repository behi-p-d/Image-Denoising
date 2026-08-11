import time

import torch
from config import DEVICE

from torchmetrics.image import (
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
)


psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(DEVICE)

ssim_metric = StructuralSimilarityIndexMeasure(
    data_range=1.0
).to(DEVICE)


def calculate_psnr(prediction, target):


    value = psnr_metric(prediction, target)

    return value.item()



def calculate_ssim(prediction, target):


    value = ssim_metric(prediction, target)

    return value.item()



def count_parameters(model):


    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )



def measure_inference_time(
    model,
    image,
    device=DEVICE,
):


    model.eval()

    image = image.to(device)
    model = model.to(device)

    with torch.no_grad():

        start = time.perf_counter()

        _ = model(image)

        end = time.perf_counter()

    return end - start



