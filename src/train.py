import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys
import argparse

# Add project root to path for relative imports if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.resnet_bit import Model
from src.attacks.base_attacks import AdversarialAttack
from src.utils.data_loader import get_dataloaders

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = "./checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)

def train_epoch(model, train_loader, optimizer, criterion, epoch, attacker=None):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(DEVICE), target.to(DEVICE)
        
        if attacker:
            model.eval()
            with torch.enable_grad():
                data = attacker.pgd(data, target)
            model.train()
            
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
    print(f'Train Epoch: {epoch} | Loss: {running_loss/len(train_loader):.4f} | Acc: {100.0 * correct / total:.2f}%')

def evaluate(model, test_loader, criterion, attacker=None):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            if attacker:
                with torch.enable_grad():
                    data = attacker.fgsm(data, target)
            output = model(data)
            test_loss += criterion(output, target).item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
    prefix = "Adversarial" if attacker else "Clean"
    print(f'{prefix} Test Accuracy: {100.0 * correct / total:.2f}%')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ResNet-BiT model on CIFAR-10.")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.0001, help="Base learning rate")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for training")
    
    args = parser.parse_args()

    print("Initializing Custom ResNetV2-BiT model...")
    model = Model(num_classes=10).to(DEVICE)
    
    train_loader, test_loader = get_dataloaders(batch_size=args.batch_size)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    
    # Adversarial Training setup (optional/default)
    attacker = AdversarialAttack(model, epsilon=8/255.0, alpha=2/255.0, iters=10)
    
    print(f"Starting Training for {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        train_epoch(model, train_loader, optimizer, criterion, epoch, attacker=attacker)
        evaluate(model, test_loader, criterion)
        evaluate(model, test_loader, criterion, attacker=attacker)
        
        if epoch % 5 == 0:
            model.save(f"{SAVE_DIR}/resnet_bit_epoch_{epoch}.pth")
