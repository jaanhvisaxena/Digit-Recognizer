import sys

import numpy as np
from PIL import Image
from mnist_loader import load_data_wrapper, vectorized_result
from network import Network, sigmoid, sigmoid_prime, save_pretrained_network, get_pretrained_network


def preprocess_image(image_path):
    """Load and preprocess an image to fit the MNIST format."""
    try:
        img = Image.open(image_path).convert('L')  # Convert to grayscale
        img = img.resize((28, 28), Image.LANCZOS)  # Resize to 28x28 with LANCZOS filter
        img = np.array(img).astype(np.float32)  # Convert to numpy array
        img = (255.0 - img) / 255.0  # Invert colors and normalize to 0-1
        img = img.flatten().reshape((784, 1))  # Flatten and reshape to (784, 1)
        return img
    except Exception as e:
        print(f"Error processing image: {e}")
        sys.exit(1)



def test_image(network, image_path):
    """Test a single image using the neural network."""
    img = preprocess_image(image_path)
    output = network.feedforward(img)
    predicted_digit = np.argmax(output)
    return predicted_digit

if __name__ == "__main__":
    # Load the trained network (you might need to train it first if no pre-trained data is available)
    network = Network(sizes=[784, 16, 16, 10])
    # Load pre-trained weights and biases if available, else train the network
    try:
        network = get_pretrained_network()
    except FileNotFoundError:
        print("Pre-trained network not found. Training a new network...")
        training_data, validation_data, test_data = load_data_wrapper()
        network.SGD(training_data, 30, 10, 3.0)
        save_pretrained_network()

    # Test on a custom image
    image_path = "Seven_full.jpg"  # Provide the path to your image here
    predicted_digit = test_image(network, image_path)
    print(f"The predicted digit is: {predicted_digit}")
