
import time
import torch
from config import DEVICE
from metrics import (
    calculate_psnr,
    calculate_ssim,
    measure_inference_time,
)


def evaluate_model(
    model,
    dataloader,
    device=DEVICE,
):

    model = model.to(device)
    model.eval()

    total_psnr = 0.0
    total_ssim = 0.0
    total_time = 0.0

    num_images = 0


    with torch.no_grad():

        for noisy_images, clean_images in dataloader:

            noisy_images = noisy_images.to(device)
            clean_images = clean_images.to(device)


            outputs = model(noisy_images)


            psnr = calculate_psnr(
                outputs,
                clean_images,
            )


            ssim = calculate_ssim(
                outputs,
                clean_images,
            )


           

            batch_size = noisy_images.size(0)

            total_psnr += psnr * batch_size
            total_ssim += ssim * batch_size
           

            num_images += batch_size


    parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    return {

        "PSNR": total_psnr / num_images,
        "SSIM": total_ssim / num_images,
        "Parameters": parameters,
    }