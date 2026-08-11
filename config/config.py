
import random

import numpy as np
import torch
from pathlib import Path


# paths

IMAGE_ROOT = Path(r"your_path\Project\data\images")
TRAIN_DIR = IMAGE_ROOT / "train"
VAL_DIR = IMAGE_ROOT / "val"
TEST_DIR = IMAGE_ROOT / "test"




# results paths

RESULTS_DIR = Path(r"your_path\Project\results")
MODELS_DIR = RESULTS_DIR / "models"
PLOTS_DIR = RESULTS_DIR / "plots"
IMAGES_DIR = RESULTS_DIR / "images"
TABLES_DIR = RESULTS_DIR / "tables"
LOGS_DIR = RESULTS_DIR / "logs"
HISTORY_DIR = RESULTS_DIR / "histories"

HISTORY_DIR.mkdir(
    parents=True,
    exist_ok=True,
)




# reproducibility

SEED = 42




# device
# so we could run everthing on both cpu and gpu

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)





# Dataset

PATCH_SIZE = 128

PATCHES_PER_IMAGE = 10





# training

BATCH_SIZE = 32

LEARNING_RATE = 1e-3

EPOCHS = 50

NUM_WORKERS = 0

PIN_MEMORY = False





# loss

ALPHA = 0.8



# seeds

def set_seed(seed=SEED):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed(seed)

        torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = True

        torch.backends.cudnn.benchmark = False


OPTIMIZER = "Adam"
