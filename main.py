import matplotlib.pyplot as plt
from data.mnist_dataset import load_mnist
from hopfield.hopfield_network import HopfieldNetwork
from hopfield.utils import add_noise, similarity_score


def main():
    # Load MNIST dataset
    x_train, _ = load_mnist()

    # Initialize and train the Hopfield Network
    network = HopfieldNetwork(size=28 * 28)  # Each MNIST image is 28x28 pixels
    sample_patterns = [x_train[0].flatten(), x_train[1].flatten()]  # Store some samples
    network.train(sample_patterns)

    # Test with a noisy version of a stored pattern
    test_image = x_train[0]
    noisy_test_image = add_noise(test_image, noise_level=0.3).flatten()
    recalled_image = network.recall(noisy_test_image)

    # Display the images
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(test_image, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Noisy Image")
    plt.imshow(noisy_test_image.reshape(28, 28), cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Recalled Image")
    plt.imshow(recalled_image.reshape(28, 28), cmap="gray")
    plt.axis("off")

    plt.show()

    # Calculate and print similarity score
    original_image = x_train[0].flatten()
    similarity = similarity_score(original_image, recalled_image)
    print(f"Similarity Score: {similarity:.2f}%")


if __name__ == "__main__":
    main()
