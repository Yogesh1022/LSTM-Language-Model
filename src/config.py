"""
Configuration settings for LSTM Language Model training
"""

def get_config(model_type='small'):
    """
    Get model configuration based on model type
    
    Args:
        model_type: 'small', 'medium', or 'large'
    
    Returns:
        dict: Configuration parameters
    """
    
    # Base configuration
    base_config = {
        'data_path': 'dataset/Pride_and_Prejudice-Jane_Austen.txt',
        'vocab_path': 'vocab/vocab.pkl',
        'model_save_dir': 'models/',
        'results_dir': 'results/',
        
        # Data parameters
        'seq_length': 35,
        'min_freq': 2,
        'batch_size': 64,
        'train_ratio': 0.8,
        'val_ratio': 0.1,
        'num_workers': 0,
        
        # Training parameters
        'num_epochs': 20,
        'learning_rate': 0.0005,
        'grad_clip': 5.0,
        'patience': 5,
        'save_every': 5,
        
        # Generation parameters
        'gen_length': 50,
        'temperature': 1.0,
    }
    
    # Model-specific configurations
    model_configs = {
        'small': {
            'embedding_dim': 128,
            'hidden_dim': 256,
            'num_layers': 1,
            'dropout': 0.3,
        },
        'medium': {
            'embedding_dim': 256,
            'hidden_dim': 512,
            'num_layers': 2,
            'dropout': 0.4,
        },
        'large': {
            'embedding_dim': 512,
            'hidden_dim': 1024,
            'num_layers': 3,
            'dropout': 0.6,
        }
    }
    
    if model_type not in model_configs:
        raise ValueError(f"Unknown model type: {model_type}. Choose from {list(model_configs.keys())}")
    
    # Merge base config with model-specific config
    config = {**base_config, **model_configs[model_type]}
    config['model_type'] = model_type
    
    return config
