from mvit_model import MViT 
from mvit.config.defaults import get_cfg 
from PIL import Image 
import requests 
import matplotlib.pyplot as plt 

def count_parameters(model):
    """
    Counts the total number of trainable parameters in a given model.

    Args:
        model: The PyTorch model for which to count parameters.

    Returns:
        The total number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


url = "https://assets-c4akfrf5b4d3f4b7.z01.azurefd.net/assets/2024/04/BMDataViz_661fb89f3845e.png" 
image = Image.open(requests.get(url, stream=True).raw) 


config = get_cfg()
model = MViT(config)
pretrained_weights = torch.load("MViTv2_T_in1k.pyth")

model.load_state_dict(pretrained_weights['model']) 
plt.imshow(image)
plt.show()
print(f"Total trainable parameters: {count_parameters(model)}")