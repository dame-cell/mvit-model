import torch
from mvit_model import MViT
from mvit.config.defaults import get_cfg
from PIL import Image
import requests
import matplotlib.pyplot as plt
import argparse

def count_parameters(model):
    """
    Counts the total number of trainable parameters in a given model.

    Args:
        model: The PyTorch model for which to count parameters.

    Returns:
        The total number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main(args):
    # Load the image from URL
    url = "https://assets-c4akfrf5b4d3f4b7.z01.azurefd.net/assets/2024/04/BMDataViz_661fb89f3845e.png"
    image = Image.open(requests.get(url, stream=True).raw)

    # Get configuration
    config = get_cfg()

    # Instantiate the MViT model using the config
    model = MViT(config)

    # Load the pre-trained weights from the path provided in args
    pretrained_weights = torch.load(args.path_to_model)

    # Load the weights into the model's state_dict
    model.load_state_dict(pretrained_weights['model'])

    # Set the model to evaluation mode (if you're doing inference)
    model.eval()

    # Display the image
    plt.imshow(image)
    plt.show()

    # Print the total number of trainable parameters
    print(f"Total trainable parameters: {count_parameters(model)}")

if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser(description="MViT Model Loader and Image Display")
    parser.add_argument('--path_to_model', type=str, required=True, help="Path to the pre-trained model checkpoint (e.g., MViTv2_T_in1k.pyth)")

    ##Parse the arguments
    args = parser.parse_args()

    # Run the main function
    main(args)
