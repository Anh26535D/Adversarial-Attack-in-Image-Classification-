import torch
import torch.nn as nn
import timm

class Model(nn.Module):
    def __init__(self, model_name='resnetv2_50x1_bit', num_classes=10):
        super().__init__()
        
        # Load the base model
        self.model = timm.create_model(model_name, pretrained=True)

        # Modify stem: 3x3 conv with stride 1, and remove pooling for CIFAR-10 (32x32)
        self.model.stem.conv = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.stem.pool = nn.Identity() 

        # Freeze most parameters
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze stem.conv, stage 3, and head for fine-tuning/adversarial training
        for param in self.model.stem.conv.parameters():
            param.requires_grad = True

        for param in self.model.stages[3].parameters():
            param.requires_grad = True

        # Modify Head to match num_classes
        in_channels = self.model.head.fc.in_channels 
        self.model.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(in_channels, num_classes)
        )
        
        for param in self.model.head.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.model(x)

    def save(self, path):
        """Save model weights to file."""
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)
        print(f"Model saved to {path}")

    @classmethod
    def load(cls, path, device='cpu', **kwargs):
        """Load model weights from file (.pth or .safetensors)."""
        model = cls(**kwargs)
        
        # Decide device
        if torch.cuda.is_available() and device == 'cuda':
            device_obj = torch.device('cuda')
        else:
            device_obj = torch.device('cpu')
            
        print(f"Loading weights from {path}...")
        
        try:
            # Check for safetensors support and format
            if path.endswith('.safetensors'):
                from safetensors.torch import load_file
                state_dict = load_file(path, device=str(device_obj))
                model.load_state_dict(state_dict)
            else:
                # Standard PyTorch load
                state_dict = torch.load(path, map_location=device_obj)
                model.load_state_dict(state_dict)
                
            model.to(device_obj)
            model.eval()
            print(f"Model successfully loaded from {path} on {device_obj}")
            
        except Exception as e:
            print(f"\nError loading model from {path}: {e}")
            if "failed finding central directory" in str(e) or "magic number" in str(e):
                print("Tip: The model file may be corrupted or in an unexpected format.")
                print("If you downloaded it recently, ensure the download completed successfully.")
            elif "No module named 'safetensors'" in str(e):
                print("Tip: Install 'safetensors' to load .safetensors models: pip install safetensors")
            
            # Re-raise to let the script exit if loading fails
            raise e
            
        return model
