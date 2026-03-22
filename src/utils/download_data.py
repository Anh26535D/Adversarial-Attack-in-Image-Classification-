import os
import torch
from torchvision import datasets

def download_cifar10(data_dir='./data'):
    """Utility to download CIFAR-10 using torchvision."""
    print(f"Checking for CIFAR-10 in {data_dir}...")
    
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        
    # Official torchvision datasets.CIFAR10 handles downloading and integrity checks
    train_set = datasets.CIFAR10(root=data_dir, train=True, download=True)
    test_set = datasets.CIFAR10(root=data_dir, train=False, download=True)
    
    print(f"\nDownload complete.")
    print(f"Training samples: {len(train_set)}")
    print(f"Testing samples:  {len(test_set)}")
    print(f"Data stored in:   {os.path.abspath(data_dir)}")

if __name__ == "__main__":
    download_cifar10()
