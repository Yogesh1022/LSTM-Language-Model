import torch
import sys
from pathlib import Path
import os

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from dataset import load_and_preprocess_data, create_dataloaders
from model import LSTMLanguageModel
from train import LanguageModelTrainer
from evaluate import ModelEvaluator, create_comparison_report
from utils import (plot_training_curves, plot_model_comparison, 
                   plot_perplexity_comparison)
from config import get_config

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    # Load and preprocess data (once for all models)
    print("\n" + "=" * 70)
    print("STEP 1: DATA PREPARATION")
    print("=" * 70)
    
    config = get_config('small')  # Use any config for data loading
    dataset, vocab, vocab_size = load_and_preprocess_data(
        config['data_path'],
        config['vocab_path'],
        config['seq_length'],
        config['min_freq']
    )
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        dataset,
        config['train_ratio'],
        config['val_ratio'],
        config['batch_size'],
        config['num_workers']
    )
    
    # Train all three models with resume capability
    model_types = ['small', 'medium', 'large']
    all_results = {}
    all_metrics = {}
    
    # Check for existing models
    print("\n" + "=" * 70)
    print("CHECKING FOR EXISTING MODELS")
    print("=" * 70)
    
    existing_models = []
    missing_models = []
    
    for model_type in model_types:
        config = get_config(model_type)
        checkpoint_paths = [
            os.path.join(config['model_save_dir'], f"{model_type}_model_best.pt"),
            os.path.join(config['model_save_dir'], f"{model_type}_best.pt")
        ]
        
        found = False
        for checkpoint_path in checkpoint_paths:
            if os.path.exists(checkpoint_path):
                try:
                    checkpoint = torch.load(checkpoint_path, map_location='cpu')
                    epoch = checkpoint.get('epoch', 'N/A')
                    val_loss = checkpoint.get('val_loss', 0)
                    print(f"✅ {model_type:8} - Found at {checkpoint_path}")
                    print(f"   Epoch {epoch}, Val Loss: {val_loss:.4f}")
                    existing_models.append((model_type, checkpoint_path))
                    found = True
                    break
                except:
                    print(f"⚠️  {model_type:8} - Corrupted checkpoint")
        
        if not found:
            print(f"❌ {model_type:8} - Not found (will train)")
            missing_models.append(model_type)
    
    print("=" * 70)
    print(f"\n📊 Summary:")
    print(f"   ✅ Already trained: {len(existing_models)} model(s)")
    print(f"   ❌ Need to train: {len(missing_models)} model(s)")
    print(f"\n🎯 Training Plan:")
    print(f"   SKIP:  {[m[0] for m in existing_models]}")
    print(f"   TRAIN: {missing_models}")
    print("=" * 70)
    
    # Load existing models
    if existing_models:
        print("\n" + "=" * 70)
        print(f"⏭️  LOADING {len(existing_models)} EXISTING MODEL(S)")
        print("=" * 70)
        
        for model_type, checkpoint_path in existing_models:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            all_results[model_type] = {
                'train_losses': checkpoint.get('train_losses', []),
                'val_losses': checkpoint.get('val_losses', []),
                'val_perplexities': checkpoint.get('val_perplexities', []),
                'best_epoch': checkpoint.get('epoch', 0),
                'best_val_loss': checkpoint.get('val_loss', 0),
            }
            print(f"   ✅ {model_type}: Loaded successfully")
    
    # Train only missing models
    if missing_models:
        print("\n" + "=" * 70)
        print(f"🚀 TRAINING {len(missing_models)} MODEL(S)")
        print("=" * 70)
    
    for idx, model_type in enumerate(missing_models, 1):
        print("\n" + "=" * 70)
        print(f"TRAINING {model_type.upper()} MODEL ({idx}/{len(missing_models)})")
        print("=" * 70)
        
        # Get model-specific config
        config = get_config(model_type)
        
        # Create model
        model = LSTMLanguageModel(
            vocab_size=vocab_size,
            embedding_dim=config['embedding_dim'],
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers'],
            dropout=config['dropout']
        ).to(device)
        
        print(f"\n{model_type.upper()} Model Architecture:")
        print(f"  Embedding dim: {config['embedding_dim']}")
        print(f"  Hidden dim: {config['hidden_dim']}")
        print(f"  Num layers: {config['num_layers']}")
        print(f"  Dropout: {config['dropout']}")
        print(f"  Total parameters: {model.count_parameters():,}")
        
        # Train model
        trainer = LanguageModelTrainer(
            model, train_loader, val_loader, config, device
        )
        results = trainer.train()
        all_results[model_type] = results
        
        # Plot training curves
        plot_training_curves(
            results['train_losses'],
            results['val_losses'],
            model_type.capitalize(),
            save_path=f"results/{model_type}_training_curves.png"
        )
        
        print(f"\n✓ {model_type.upper()} model training complete!")
    
    if not missing_models:
        print("\n" + "=" * 70)
        print("🎉 ALL MODELS ALREADY TRAINED!")
        print("=" * 70)
        print("To retrain, delete checkpoint files in models/ folder")
    
    # Compare all models
    print("\n" + "=" * 70)
    print("STEP 3: MODEL COMPARISON")
    print("=" * 70)
    
    plot_model_comparison(
        all_results,
        metric='val_losses',
        save_path='results/model_comparison_loss.png'
    )
    
    plot_perplexity_comparison(
        all_results,
        save_path='results/model_comparison_perplexity.png'
    )
    
    # Evaluate all models on test set
    print("\n" + "=" * 70)
    print("STEP 4: TEST SET EVALUATION")
    print("=" * 70)
    
    for model_type in model_types:
        print(f"\n{model_type.upper()} Model:")
        
        config = get_config(model_type)
        
        # Find best model checkpoint (try both naming conventions)
        checkpoint_paths = [
            os.path.join(config['model_save_dir'], f"{model_type}_model_best.pt"),
            os.path.join(config['model_save_dir'], f"{model_type}_best.pt")
        ]
        
        checkpoint_path = None
        for path in checkpoint_paths:
            if os.path.exists(path):
                checkpoint_path = path
                break
        
        if not checkpoint_path:
            print(f"   ⚠️  Model not found! Skipping evaluation.")
            continue
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        model = LSTMLanguageModel(
            vocab_size=vocab_size,
            embedding_dim=config['embedding_dim'],
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers'],
            dropout=config['dropout']
        ).to(device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Evaluate
        evaluator = ModelEvaluator(model, device)
        test_metrics = evaluator.evaluate_on_dataset(test_loader, "Test")
        
        # Get final training loss (handle both cases)
        train_losses = all_results[model_type].get('train_losses', [])
        final_train_loss = train_losses[-1] if train_losses else checkpoint.get('val_loss', 0)
        
        all_metrics[model_type] = {
            'train': {
                'final_loss': final_train_loss
            },
            'val': {
                'loss': checkpoint['val_loss'],
                'perplexity': checkpoint['val_perplexity'],
            },
            'test': test_metrics,
            'best_epoch': all_results[model_type]['best_epoch'],
            'total_epochs': len(train_losses) if train_losses else checkpoint.get('epoch', 0),
        }
    
    # Create comparison report
    create_comparison_report(
        all_metrics,
        'results/final_comparison_report.json'
    )
    
    # Text generation examples
    print("\n" + "=" * 70)
    print("STEP 5: TEXT GENERATION EXAMPLES")
    print("=" * 70)
    
    from generate import generate_multiple_samples
    
    start_texts = [
        "it is a truth",
        "elizabeth was",
        "mr darcy"
    ]
    
    # Use the best model for generation
    best_model_type = min(
        all_metrics.items(),
        key=lambda x: x[1]['test']['loss']
    )[0]
    
    print(f"\nUsing {best_model_type.upper()} model for text generation")
    
    config = get_config(best_model_type)
    
    # Find checkpoint (try both naming conventions)
    checkpoint_paths = [
        os.path.join(config['model_save_dir'], f"{best_model_type}_model_best.pt"),
        os.path.join(config['model_save_dir'], f"{best_model_type}_best.pt")
    ]
    
    checkpoint_path = None
    for path in checkpoint_paths:
        if os.path.exists(path):
            checkpoint_path = path
            break
    
    if not checkpoint_path:
        print(f"⚠️  Best model not found! Skipping text generation.")
        return
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model = LSTMLanguageModel(
        vocab_size=vocab_size,
        embedding_dim=config['embedding_dim'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        dropout=config['dropout']
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    generated_samples = generate_multiple_samples(
        model, vocab, start_texts,
        max_length=30,
        temperature=0.8,
        num_samples=3,
        device=device
    )
    
    # Save generated text
    import json
    with open('results/generated_samples.json', 'w') as f:
        json.dump(generated_samples, f, indent=4)
    
    print("\n" + "=" * 70)
    print("ALL TRAINING AND EVALUATION COMPLETE!")
    print("=" * 70)
    print("\n✓ Results saved to: results/")
    print("✓ All models saved to: models/")
    print("\nSummary:")
    for model_type in model_types:
        metrics = all_metrics[model_type]
        print(f"\n{model_type.upper()}:")
        print(f"  Best Epoch: {metrics['best_epoch']}/{metrics['total_epochs']}")
        print(f"  Val Loss: {metrics['val']['loss']:.4f}")
        print(f"  Val Perplexity: {metrics['val']['perplexity']:.2f}")
        print(f"  Test Loss: {metrics['test']['loss']:.4f}")
        print(f"  Test Perplexity: {metrics['test']['perplexity']:.2f}")

if __name__ == "__main__":
    main()
