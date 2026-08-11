
import torch


def add_gaussian_noise(image, mean=0.0, std_range=(0.05, 0.15)):


    noisy_image = image.clone()

    # sample a random std for each image 
    std = torch.empty(1).uniform_(
        std_range[0],
        std_range[1]
    ).item()

    noise = torch.randn_like(noisy_image) * std + mean

    noisy_image = noisy_image + noise

    noisy_image = torch.clamp(noisy_image, 0.0, 1.0)
    # the pixel values should be in the valid range (0,1)
    return noisy_image






def add_salt_pepper_noise(image, amount_range=(0.01, 0.05)):


    noisy_image = image.clone()

    # randomly choose number of pixels to be affected 
    amount = torch.empty(1).uniform_(
        amount_range[0],
        amount_range[1]
    ).item()

    # create one random mask per spatial pixel
    random_matrix = torch.rand(
        (1, image.shape[1], image.shape[2]),
        device=image.device
    )

    salt_mask = random_matrix < (amount / 2)

    pepper_mask = (
        (random_matrix >= (amount / 2))
        &
        (random_matrix < amount)
    )

    noisy_image[salt_mask.expand_as(noisy_image)] = 1.0
    noisy_image[pepper_mask.expand_as(noisy_image)] = 0.0

    return noisy_image





def add_speckle_noise(image, std_range=(0.05, 0.2)):


    noisy_image = image.clone()

    # randomly sample the standard deviation
    std = torch.empty(1).uniform_(
        std_range[0],
        std_range[1]
    ).item()

    # generate multiplicative Gaussian noise
    noise = torch.randn_like(noisy_image) * std

    noisy_image = noisy_image + noisy_image * noise

    noisy_image = torch.clamp(noisy_image, 0.0, 1.0)

    return noisy_image





def add_mixed_noise(image):


    noisy_image = image.clone()

    noise_functions = [
        add_gaussian_noise,
        add_salt_pepper_noise,
        add_speckle_noise,
    ]

    # decide how many noise types to apply
    probabilities = torch.tensor([0.4, 0.4, 0.2])

    num_noises = torch.multinomial(
        probabilities,
        num_samples=1
    ).item() + 1

    # randomly select noise functions
    permutation = torch.randperm(len(noise_functions))

    selected_functions = [
        noise_functions[i]
        for i in permutation[:num_noises]
    ]

    # apply them sequentially
    for noise_function in selected_functions:
        noisy_image = noise_function(noisy_image)

    return noisy_image

