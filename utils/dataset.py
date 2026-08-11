from pathlib import Path

from PIL import Image

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms

from noise import add_mixed_noise
from config import BATCH_SIZE, NUM_WORKERS, PIN_MEMORY, PATCH_SIZE, PATCHES_PER_IMAGE



class BSD500Dataset(Dataset):

    # BSD500 dataset 
    # 200 training, 200 val, 100 test 
    
    def __init__(
        self,
        image_dir,
        patch_size=PATCH_SIZE,
        patches_per_image=PATCHES_PER_IMAGE,
        use_patches=True,
        add_noise=False,
        noise_transform=None,
        transform=None,
        
    ):

        self.image_dir = Path(image_dir)
        self.patch_size = patch_size
        self.use_patches = use_patches
        self.add_noise = add_noise

        self.noise_transform = (
           noise_transform
           if noise_transform is not None
           else add_mixed_noise
        )
        self.patches_per_image = patches_per_image

        self.image_dir = Path(image_dir)

        if not self.image_dir.exists():
            raise FileNotFoundError(f"Folder not found: {self.image_dir}")

        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

        self.image_paths = sorted(
            file
            for file in self.image_dir.iterdir()
            if file.suffix.lower() in valid_extensions
        )

        if len(self.image_paths) == 0:
            raise RuntimeError(f"No images found in {self.image_dir}")

        if transform is None:

            transform_list = []

            if self.use_patches:
                transform_list.append(
                    transforms.RandomCrop((patch_size, patch_size))
                )

            transform_list.append(transforms.ToTensor())

            self.transform = transforms.Compose(transform_list)

        else:
            self.transform = transform

    def __len__(self):
        return len(self.image_paths) * self.patches_per_image

    def __getitem__(self, idx):

        
        image_index = idx // self.patches_per_image
        image = Image.open(self.image_paths[image_index]).convert("RGB")


        clean_image = self.transform(image)

        if self.add_noise:
            noisy_image = self.noise_transform(clean_image)
            return noisy_image, clean_image

        return clean_image


    
def create_dataloaders(
    train_dataset,
    val_dataset,
    test_dataset,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
):


    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader        
    


    