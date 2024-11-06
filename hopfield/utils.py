import numpy as np


def add_noise(image, noise_level=0.3):
    # Flatten the image to a 1D array
    noisy_image = image.flatten()

    # Randomly flip a percentage of pixels based on noise level
    n_pixels = len(noisy_image)

    noise = np.random.choice([1, -1], size=n_pixels, p=[1 - noise_level, noise_level])
    noisy_image = noisy_image * noise
    
    # Reshape back to the original image size
    return noisy_image.reshape(image.shape)


def similarity_score(original, recalled):
    # Compare pixel values
    matches = np.sum(original == recalled)
    total_pixels = original.size
    return (matches / total_pixels) * 100  # Convert to percentage
