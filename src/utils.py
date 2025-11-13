import matplotlib.pyplot as plt
import numpy as np
import os

def plot_training_curves(train_losses, val_losses, model_name, save_path=None):
    """Plot training and validation loss curves"""
    fig, ax = plt.subplots(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    
    ax.plot(epochs, train_losses, 'b-', label='Training Loss', 
           linewidth=2, marker='o', markersize=4)
    ax.plot(epochs, val_losses, 'r-', label='Validation Loss', 
           linewidth=2, marker='s', markersize=4)
    
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax.set_title(f'{model_name} - Training Curves', 
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved: {save_path}")
    
    plt.show()


def plot_model_comparison(results_dict, metric='val_losses', save_path=None):
    """Compare training curves across models"""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    for idx, (model_name, results) in enumerate(results_dict.items()):
        epochs = range(1, len(results[metric]) + 1)
        ax.plot(epochs, results[metric], 
               color=colors[idx % len(colors)],
               label=model_name.capitalize(),
               linewidth=2.5, marker='o', markersize=5)
    
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Validation Loss', fontsize=12, fontweight='bold')
    ax.set_title('Model Comparison - Validation Loss', 
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved: {save_path}")
    
    plt.show()


def plot_perplexity_comparison(results_dict, save_path=None):
    """Plot perplexity comparison"""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    for idx, (model_name, results) in enumerate(results_dict.items()):
        epochs = range(1, len(results['val_perplexities']) + 1)
        ax.plot(epochs, results['val_perplexities'],
               color=colors[idx % len(colors)],
               label=model_name.capitalize(),
               linewidth=2.5, marker='s', markersize=5)
    
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Perplexity', fontsize=12, fontweight='bold')
    ax.set_title('Model Comparison - Validation Perplexity',
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Perplexity comparison plot saved: {save_path}")
    
    plt.show()
