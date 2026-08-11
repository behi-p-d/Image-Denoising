

import matplotlib.pyplot as plt
from pathlib import Path


import numpy as np
import torch




def tensor_to_numpy(tensor):

    
    # convert a PyTorch tensor to a numpy array.
    # if the input is already anumpy array, it is returned unchanged.
    if torch.is_tensor(tensor):
        tensor = tensor.detach().cpu().numpy()

    return tensor


def tensor_to_image(image):
    
    # convert a tensor or numpy array into an image compatible with matplotlib
    image = tensor_to_numpy(image)

    if image.ndim == 3:
        image = image.transpose(1, 2, 0)

    return np.clip(image, 0, 1)


def plot_training_curves(
    histories,
    labels,
    title="Training Loss",
    figsize=(8, 5),
    save_path=None,
):


    
    # input validation
   
    if len(histories) != len(labels):
        raise ValueError(
            "histories and labels must have the same length."
        )

    
    # create figure
    
    plt.figure(figsize=figsize)


   
    # plot each training curve
    
    for history, label in zip(histories, labels):

        if "train_loss" not in history:
            raise KeyError(
                f"'train_loss' not found in history for '{label}'."
            )

        plt.plot(
            history["train_loss"],
            linewidth=2,
            label=label,
        )

    
    # figure formatting
    
    plt.title(title)

    plt.xlabel("Epoch")

    plt.ylabel("Training Loss")

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    
    # save figure
    
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
        )

   
    # display figure
    
    plt.show()



def plot_validation_curves(
    histories,
    labels,
    title="Validation Loss",
    figsize=(8, 5),
    save_path=None,
):


    
    # input validation
    
    if len(histories) != len(labels):
        raise ValueError(
            "histories and labels must have the same length."
        )

    
    # create figure
   
    plt.figure(figsize=figsize)

    
    # plot each validation curve
    
    for history, label in zip(histories, labels):

        if "val_loss" not in history:
            raise KeyError(
                f"'val_loss' not found in history for '{label}'."
            )

        epochs = range(1, len(history["val_loss"]) + 1)

        plt.plot(
            epochs,
            history["val_loss"],
            linewidth=2,
            label=label,
        )

    
    # figure formatting
    
    plt.title(title)

    plt.xlabel("Epoch")

    plt.ylabel("Validation Loss")

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    
    # save figure
   
    if save_path is not None:

        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
        )

    
    # display figure
   
    plt.show()






def show_denoising_results(
    clean_image,
    noisy_image,
    predictions,
    prediction_titles,
    figsize=(18, 4),
    save_path=None,
):


   
    # input validation
    
    if len(predictions) != len(prediction_titles):
        raise ValueError(
            "predictions and prediction_titles "
            "must have the same length."
        )

   
    # prepare images and titles
    
    images = [
        clean_image,
        noisy_image,
        *predictions,
    ]

    titles = [
        "Original",
        "Noisy",
        *prediction_titles,
    ]

    num_images = len(images)

    
    # create figure
   
    plt.figure(figsize=figsize)

   
    # display images
    
    for i, (image, title) in enumerate(
        zip(images, titles),
        start=1,
    ):


        plt.subplot(1, num_images, i)

        plt.imshow(tensor_to_image(image))

        plt.title(title)

        plt.axis("off")

    
    # layout
   
    plt.tight_layout()

    
    # save figure
    
    if save_path is not None:

        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
        )

    
    # display figure
    
    plt.show()




def plot_error_maps(
    clean_image,
    predictions,
    prediction_titles,
    figsize=(12, 4),
    save_path=None,
):

    plt.figure(figsize=figsize)



    num_images = len(predictions)

    clean_image = tensor_to_numpy(clean_image)

    for i, (prediction, title) in enumerate(
        zip(predictions, prediction_titles),
        start=1,
    ):

        prediction = tensor_to_numpy(prediction)

        error = np.abs(clean_image - prediction)

        error = error.mean(axis=0)

        plt.subplot(1, num_images, i)

        plt.imshow(
            error,
            cmap="hot",
        )

        plt.title(f"{title} Error")

        plt.axis("off")

        plt.colorbar(
            fraction=0.046,
            pad=0.04,
        )



    plt.tight_layout()

    if save_path is not None:

        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
        )

    plt.show()