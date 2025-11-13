import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from datetime import datetime
import os

class LanguageModelTrainer:
    """Trainer for language model"""
    
    def __init__(self, model, train_loader, val_loader, config, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
        
        self.train_losses = []
        self.val_losses = []
        self.val_perplexities = []
        
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.best_epoch = 0
    
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, (x, y) in enumerate(progress_bar):
            x, y = x.to(self.device), y.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            logits, _ = self.model(x)
            
            # Reshape for loss calculation
            loss = self.criterion(
                logits.view(-1, self.model.vocab_size),
                y.view(-1)
            )
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config['grad_clip']
            )
            
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss
    
    def evaluate(self):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for x, y in self.val_loader:
                x, y = x.to(self.device), y.to(self.device)
                
                logits, _ = self.model(x)
                
                loss = self.criterion(
                    logits.view(-1, self.model.vocab_size),
                    y.view(-1)
                )
                
                total_loss += loss.item()
        
        avg_loss = total_loss / len(self.val_loader)
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        return avg_loss, perplexity
    
    def save_checkpoint(self, epoch, val_loss, val_ppl):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'val_perplexity': val_ppl,
            'config': self.config,
        }
        
        os.makedirs(self.config['model_save_dir'], exist_ok=True)
        filepath = os.path.join(
            self.config['model_save_dir'],
            f"{self.config['model_type']}_epoch_{epoch}.pt"
        )
        torch.save(checkpoint, filepath)
        return filepath
    
    def train(self):
        print("\n" + "=" * 70)
        print(f"TRAINING {self.config['model_type'].upper()} MODEL")
        print("=" * 70)
        print(f"Model parameters: {self.model.count_parameters():,}")
        
        for epoch in range(1, self.config['num_epochs'] + 1):
            # Train
            train_loss = self.train_epoch(epoch)
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss, val_ppl = self.evaluate()
            self.val_losses.append(val_loss)
            self.val_perplexities.append(val_ppl)
            
            print(f"\nEpoch {epoch}/{self.config['num_epochs']}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Val Perplexity: {val_ppl:.2f}")
            
            # Save checkpoint
            if epoch % self.config['save_every'] == 0:
                filepath = self.save_checkpoint(epoch, val_loss, val_ppl)
                print(f"  Checkpoint saved: {filepath}")
            
            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                self.patience_counter = 0
                
                # Save best model (use .pt extension, but save as _model_best for consistency)
                best_path = os.path.join(
                    self.config['model_save_dir'],
                    f"{self.config['model_type']}_model_best.pt"
                )
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_perplexity': val_ppl,
                    'config': self.config,
                }
                torch.save(checkpoint, best_path)
                print(f"  ✓ New best model saved!")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config['patience']:
                    print(f"\nEarly stopping triggered at epoch {epoch}")
                    print(f"Best epoch was {self.best_epoch} with val loss {self.best_val_loss:.4f}")
                    break
        
        print("\n" + "=" * 70)
        print("TRAINING COMPLETE")
        print("=" * 70)
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_perplexities': self.val_perplexities,
            'best_epoch': self.best_epoch,
            'best_val_loss': self.best_val_loss,
        }
