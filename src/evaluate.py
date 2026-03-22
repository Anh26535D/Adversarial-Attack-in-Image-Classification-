import torch
import os
import sys
import pandas as pd
import argparse

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.resnet_bit import Model
from src.attacks.base_attacks import AdversarialAttack
from src.defenses.post_processing import RobustInference
from src.utils.data_loader import get_dataloaders

# Config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_MODEL_PATH = "checkpoints/resnet_bit_epoch_20.pth"

def run_quantitative_eval(model_path=DEFAULT_MODEL_PATH, epsilon=8/255.0):
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}...")
        model = Model.load(model_path, device=DEVICE, num_classes=10)
    else:
        print(f"Warning: Checkpoint '{model_path}' not found. Evaluating base model.")
        model = Model(num_classes=10).to(DEVICE)
    
    _, test_loader = get_dataloaders(batch_size=128)
    
    attacker = AdversarialAttack(model, epsilon=epsilon, iters=10)
    infer = RobustInference(model, device=DEVICE)
    
    results = {"Scenario": [], "Accuracy (%)": []}
    scenarios = [
        ("Clean", None),
        ("FGSM Attack", "fgsm"),
        ("PGD Attack", "pgd"),
        ("Smoothing Defense (PGD)", "smoothing_pgd"),
        ("Bit Depth Defense (PGD)", "bit_depth_pgd")
    ]
    
    print(f"\nRunning Robustness Evaluation (epsilon={epsilon:.4f})...")
    for label, mode in scenarios:
        correct = 0
        total = 0
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            
            if mode == "fgsm":
                data = attacker.fgsm(data, target)
            elif mode in ["pgd", "smoothing_pgd", "bit_depth_pgd"]:
                with torch.enable_grad():
                    data = attacker.pgd(data, target)
            
            defense = "smoothing" if mode == "smoothing_pgd" else ("bit_depth" if mode == "bit_depth_pgd" else None)
            pred, _ = infer.predict(data, defense=defense)
            
            total += target.size(0)
            correct += pred.eq(target).sum().item()
            
        acc = 100.0 * correct / total
        results["Scenario"].append(label)
        results["Accuracy (%)"].append(acc)
        print(f"  {label}: {acc:.2f}%")
        
    df = pd.DataFrame(results)
    print("\n--- Quantitative Evaluation Summary ---")
    print(df.to_string(index=False))
    
    results_dir = "./results"
    os.makedirs(results_dir, exist_ok=True)
    model_name = os.path.basename(model_path).replace(".pth", "")
    output_file = f"{results_dir}/eval_{model_name}.csv"
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model robustness against adversarial attacks.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH, 
                        help=f"Path to the model checkpoint (default: {DEFAULT_MODEL_PATH})")
    parser.add_argument("--epsilon", type=float, default=8/255.0, 
                        help="Maximum perturbation for attacks (default: 8/255.0)")
    
    args = parser.parse_args()
    run_quantitative_eval(model_path=args.model, epsilon=args.epsilon)

