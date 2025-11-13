import torch
import torch.nn as nn
import json
from datetime import datetime
from tqdm import tqdm
import os

class ModelEvaluator:
    """Evaluate trained models"""
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
    
    def evaluate_on_dataset(self, dataloader, dataset_name="Test"):
        self.model.eval()
        total_loss = 0
        total_tokens = 0
        
        print(f"\nEvaluating on {dataset_name} set...")
        
        with torch.no_grad():
            for x, y in tqdm(dataloader, desc=f"{dataset_name} evaluation"):
                x, y = x.to(self.device), y.to(self.device)
                
                logits, _ = self.model(x)
                
                loss = self.criterion(
                    logits.view(-1, self.model.vocab_size),
                    y.view(-1)
                )
                
                total_loss += loss.item() * y.numel()
                total_tokens += y.numel()
        
        avg_loss = total_loss / total_tokens
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        metrics = {
            'loss': avg_loss,
            'perplexity': perplexity,
            'total_tokens': total_tokens,
        }
        
        print(f"{dataset_name} Results:")
        print(f"  Loss: {avg_loss:.4f}")
        print(f"  Perplexity: {perplexity:.2f}")
        
        return metrics
    
    @staticmethod
    def save_metrics(metrics_dict, filepath):
        with open(filepath, 'w') as f:
            json.dump(metrics_dict, f, indent=4)
        print(f"Metrics saved to: {filepath}")
    
    @staticmethod
    def load_metrics(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)


def create_comparison_report(metrics_all, output_path):
    print("\n" + "=" * 70)
    print("FINAL MODEL COMPARISON")
    print("=" * 70)
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'models': metrics_all
    }
    
    best_model = min(metrics_all.items(), 
                     key=lambda x: x[1]['test']['loss'])
    
    print(f"\nBest Model: {best_model[0]}")
    print(f"  Test Loss: {best_model[1]['test']['loss']:.4f}")
    print(f"  Test Perplexity: {best_model[1]['test']['perplexity']:.2f}")
    
    print("\n" + "-" * 70)
    for model_name, metrics in metrics_all.items():
        print(f"\n{model_name}:")
        print(f"  Test Loss: {metrics['test']['loss']:.4f}")
        print(f"  Test Perplexity: {metrics['test']['perplexity']:.2f}")
    
    print("=" * 70 + "\n")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=4)
    
    print(f"Comparison report saved to: {output_path}")
    
    return report
