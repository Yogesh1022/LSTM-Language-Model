import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pickle
import os
from collections import Counter
from tqdm import tqdm

class Vocabulary:
    """Build and manage vocabulary for text data"""
    
    def __init__(self):
        self.word2idx = {
            "<PAD>": 0,
            "<UNK>": 1,
            "<SOS>": 2,
            "<EOS>": 3,
        }
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        self.word_count = Counter()
    
    def add_word(self, word):
        """Add word to vocabulary"""
        if word not in self.word2idx:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word
        self.word_count[word] += 1
    
    def build_vocab(self, text, min_freq=2):
        """Build vocabulary from text"""
        print("\nBuilding vocabulary...")
        
        # Tokenize: simple whitespace + lowercase
        words = text.lower().split()
        
        # Count all words
        for word in tqdm(words, desc="Counting words"):
            self.word_count[word] += 1
        
        print(f"Total unique words (before filtering): {len(self.word_count)}")
        
        # Add words that appear at least min_freq times
        filtered_words = 0
        for word, count in self.word_count.items():
            if count >= min_freq:
                if word not in self.word2idx:
                    idx = len(self.word2idx)
                    self.word2idx[word] = idx
                    self.idx2word[idx] = word
            else:
                filtered_words += 1
        
        print(f"Words filtered out (freq < {min_freq}): {filtered_words}")
        
        return len(self.word2idx)
    
    def encode(self, text):
        """Convert text to token indices"""
        words = text.lower().split()
        return [self.word2idx.get(w, self.word2idx["<UNK>"]) for w in words]
    
    def decode(self, indices):
        """Convert token indices back to text"""
        return [self.idx2word.get(idx, "<UNK>") for idx in indices]
    
    def save(self, filepath):
        """Save vocabulary to file"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump({
                'word2idx': self.word2idx,
                'idx2word': self.idx2word,
                'word_count': self.word_count
            }, f)
    
    @classmethod
    def load(cls, filepath):
        """Load vocabulary from file"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        vocab = cls()
        vocab.word2idx = data['word2idx']
        vocab.idx2word = data['idx2word']
        vocab.word_count = data['word_count']
        return vocab


class LanguageModelDataset(Dataset):
    """Dataset for language modeling"""
    
    def __init__(self, text, vocab, seq_length=35):
        self.vocab = vocab
        self.seq_length = seq_length
        
        # Encode the entire text
        print("\nEncoding text...")
        self.data = vocab.encode(text)
        
        print(f"Encoded text length: {len(self.data)} tokens")
        print(f"Sequence length: {seq_length}")
        print(f"Total sequences: {len(self)}")
    
    def __len__(self):
        return max(0, len(self.data) - self.seq_length)
    
    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx:idx + self.seq_length], dtype=torch.long)
        y = torch.tensor(self.data[idx + 1:idx + self.seq_length + 1], dtype=torch.long)
        
        return x, y


def load_and_preprocess_data(data_path, vocab_path, seq_length=35, min_freq=2):
    """Load data and preprocess"""
    print("=" * 60)
    print("LOADING AND PREPROCESSING DATA")
    print("=" * 60)
    
    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    print(f"Raw text size: {len(text)} characters")
    
    vocab = Vocabulary()
    vocab_size = vocab.build_vocab(text, min_freq=min_freq)
    
    print(f"\nVocabulary built!")
    print(f"Vocabulary size: {vocab_size}")
    
    vocab.save(vocab_path)
    print(f"Vocabulary saved to: {vocab_path}")
    
    dataset = LanguageModelDataset(text, vocab, seq_length=seq_length)
    
    return dataset, vocab, vocab_size


def create_dataloaders(dataset, train_ratio=0.8, val_ratio=0.1, 
                       batch_size=64, num_workers=0):
    """Split dataset and create dataloaders"""
    print("\n" + "=" * 60)
    print("CREATING DATALOADERS")
    print("=" * 60)
    
    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"Train set: {len(train_dataset)} samples")
    print(f"Val set: {len(val_dataset)} samples")
    print(f"Test set: {len(test_dataset)} samples")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    return train_loader, val_loader, test_loader
