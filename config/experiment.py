
import torch
import json 

from config import (
    DEVICE,
    LEARNING_RATE,
    MODELS_DIR,
    BATCH_SIZE,
    HISTORY_DIR,
)

from trainer import train_model



def run_experiment(
    model_class,
    criterion,
    train_loader,
    val_loader,
    epochs,
    model_name,
    loss_name,
    device=DEVICE,
):


    model = model_class().to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    save_path = MODELS_DIR / f"{model_name}_{loss_name}_best.pth"

    history, best_val_loss = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        epochs=epochs,
        save_path=save_path,
        device=device,
    )

    history_path = HISTORY_DIR / f"{model_name}_{loss_name}_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=4)


    return {
    "model": model,
    "history": history,
    "best_val_loss": best_val_loss,
    "checkpoint": save_path,
    "model_name": model_name,
    "loss_name": loss_name,
    "epochs": epochs,
    "learning_rate": LEARNING_RATE,
    "batch_size": BATCH_SIZE,
    "history_file": history_path,
}
   