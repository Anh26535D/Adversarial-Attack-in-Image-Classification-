import torch
import torch.nn as nn

def bit_depth_reduction(images, bits=3):
    """
    Defense: Bit depth reduction.
    Reduces the number of bits used to represent color values.
    """
    levels = 2 ** bits
    # Quantize
    quantized = torch.round(images * (levels - 1)) / (levels - 1)
    return quantized

def gaussian_noise_defense(images, std=0.05):
    """
    Defense: Gaussian noise.
    Adding a small amount of Gaussian noise can sometimes 'drown out' adversarial noise.
    """
    noise = torch.randn_like(images) * std
    return torch.clamp(images + noise, 0, 1)

def smoothing_defense(images, kernel_size=3):
    """
    Defense: Gaussian smoothing/blur.
    Uses a blur filter to mitigate sharp adversarial perturbations.
    """
    padding = kernel_size // 2
    avg_pool = nn.AvgPool2d(kernel_size=kernel_size, stride=1, padding=padding)
    return avg_pool(images)

class RobustInference:
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.model.eval()
        self.device = device

    def predict(self, images, defense=None, **kwargs):
        """
        Inference with optional post-processing defense.
        """
        images = images.to(self.device)
        
        if defense == 'bit_depth':
            bits = kwargs.get('bits', 3)
            images = bit_depth_reduction(images, bits=bits)
        elif defense == 'gaussian':
            std = kwargs.get('std', 0.05)
            images = gaussian_noise_defense(images, std=std)
        elif defense == 'smoothing':
            k_size = kwargs.get('kernel_size', 3)
            images = smoothing_defense(images, kernel_size=k_size)
            
        with torch.no_grad():
            outputs = self.model(images)
            _, predicted = torch.max(outputs, 1)
            
        return predicted, outputs
