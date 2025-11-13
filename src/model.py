import torch
import torch.nn as nn

class LSTMLanguageModel(nn.Module):
    """LSTM-based Language Model"""
    
    def __init__(self, vocab_size, embedding_dim, hidden_dim, 
                 num_layers, dropout=0.5):
        super(LSTMLanguageModel, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # LSTM layer
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Output layer
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x, hidden=None):
        embedded = self.dropout(self.embedding(x))
        
        if hidden is None:
            output, hidden = self.lstm(embedded)
        else:
            output, hidden = self.lstm(embedded, hidden)
        
        output = self.dropout(output)
        logits = self.fc(output)
        
        return logits, hidden
    
    def init_hidden(self, batch_size, device):
        h0 = torch.zeros(self.num_layers, batch_size, 
                        self.hidden_dim).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, 
                        self.hidden_dim).to(device)
        return (h0, c0)
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
