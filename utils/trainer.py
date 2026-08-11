
import torch
from tqdm import tqdm
from pathlib import Path
from config import DEVICE



def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device=DEVICE,
):

    model.train()

    running_loss = 0.0

    for noisy_images, clean_images in tqdm(
        dataloader,
        desc="Training",
        leave=False,
    ):

        noisy_images = noisy_images.to(device)
        clean_images = clean_images.to(device)

        optimizer.zero_grad()

        outputs = model(noisy_images)

        loss = criterion(outputs, clean_images)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(dataloader)

    return epoch_loss




def validate(
    model,
    dataloader,
    criterion,
    device=DEVICE,
):


    model.eval()

    running_loss = 0.0

    with torch.no_grad():

        for noisy_images, clean_images in tqdm(
            dataloader,
            desc="Validation",
            leave=False,
        ):

            noisy_images = noisy_images.to(device)
            clean_images = clean_images.to(device)

            outputs = model(noisy_images)

            loss = criterion(outputs, clean_images)

            running_loss += loss.item()

    epoch_loss = running_loss / len(dataloader)

    return epoch_loss




def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    epochs,
    save_path,
    device=DEVICE,
):
    
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    model.to(device)

    history = {

        "train_loss": [],

        "val_loss": []

    }

    best_val_loss = float("inf")

    for epoch in range(epochs):

        print(f"\nEpoch {epoch + 1}/{epochs}")

        train_loss = train_one_epoch(

            model,

            train_loader,

            criterion,

            optimizer,

            device,

        )

        val_loss = validate(

            model,

            val_loader,

            criterion,

            device,

        )        

        history["train_loss"].append(train_loss)

        history["val_loss"].append(val_loss)


        if val_loss < best_val_loss:

            best_val_loss = val_loss

            torch.save(

                model.state_dict(),

                save_path,

            )

            saved = "✓"

        else:

            saved = ""   

        print(

            f"Train Loss: {train_loss:.6f} | "

            f"Validation Loss: {val_loss:.6f} "

            f"{saved}"

        )            
    return history, best_val_loss       