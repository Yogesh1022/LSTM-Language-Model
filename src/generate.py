import torch
import numpy as np

def generate_text(model, vocab, start_text, max_length=50, 
                 temperature=1.0, device='cpu'):
    """Generate text using the trained model"""
    model.eval()
    
    words = start_text.lower().split()
    input_seq = [vocab.word2idx.get(w, vocab.word2idx["<UNK>"]) 
                 for w in words]
    
    generated = words.copy()
    
    with torch.no_grad():
        for _ in range(max_length):
            # Prepare input
            x = torch.tensor([input_seq], dtype=torch.long).to(device)
            
            # Get predictions
            logits, _ = model(x)
            
            # Get last token's logits
            logits = logits[0, -1, :] / temperature
            
            # Sample from distribution
            probs = torch.softmax(logits, dim=0)
            next_token = torch.multinomial(probs, 1).item()
            
            # Add to sequence
            input_seq.append(next_token)
            input_seq = input_seq[-35:]  # Keep last 35 tokens
            
            # Decode and add to generated text
            next_word = vocab.idx2word.get(next_token, "<UNK>")
            generated.append(next_word)
    
    return ' '.join(generated)


def generate_multiple_samples(model, vocab, start_texts, max_length=50, 
                             temperature=1.0, num_samples=3, device='cpu'):
    """Generate multiple samples"""
    results = {}
    
    for start_text in start_texts:
        results[start_text] = []
        print(f"\nGenerating samples for: '{start_text}'")
        
        for i in range(num_samples):
            generated = generate_text(
                model, vocab, start_text, max_length, temperature, device
            )
            results[start_text].append(generated)
            print(f"  Sample {i+1}: {generated}")
    
    return results
