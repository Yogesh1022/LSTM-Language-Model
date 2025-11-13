# LSTM Language Model - Pride and Prejudice

A complete implementation of LSTM-based language models with three different architectures (Small, Medium, Large) trained on Jane Austen's "Pride and Prejudice".

## 📁 Project Structure

```
Assignment2/
├── dataset/
│   └── Pride_and_Prejudice-Jane_Austen.txt
├── models/                    # Saved model checkpoints
├── results/                   # Training curves, metrics, comparisons
├── vocab/                     # Vocabulary files
├── src/
│   ├── __init__.py
│   ├── config.py             # Model configurations
│   ├── dataset.py            # Data loading and preprocessing
│   ├── model.py              # LSTM model architecture
│   ├── train.py              # Training logic
│   ├── evaluate.py           # Evaluation metrics
│   ├── generate.py           # Text generation
│   └── utils.py              # Plotting utilities
├── train_all_models.py       # Main training script
├── requirements.txt
└── README.md
```

## 📥 Pre-trained Models

**Download trained models from Google Drive:**

| Model | Size | Test Perplexity | Download Link |
|-------|------|-----------------|---------------|
| Small Model | ~68 MB | 4.58 | [Download small_model_best.pt](https://drive.google.com/drive/folders/1JNwQmHFHO0f_Z5guhp5MbNnaUII8-1Lr?usp=sharing) |
| Medium Model | ~68 MB | 2.04 | [Download medium_model_best.pt](https://drive.google.com/drive/folders/1IZ5mmpynoMoIMVD8qN8NDfk70o3YTgxK?usp=sharing) |
| Large Model | ~130 MB | 1.79 | [Download large_model_best.pt](https://drive.google.com/drive/folders/1CbgUJLIbKW3N_Kc7aaxhomh3jYCoVhcV?usp=sharing) |

**Instructions:**
1. Click the download link for the model you want
2. Download the `.pt` file
3. Place it in the `models/` directory
4. Use for evaluation or text generation

---

## 🚀 Quick Start

### Option 1: VS Code Jupyter Notebook (Recommended!)

**Interactive training in VS Code:**

1. Open `notebooks/training_notebook.ipynb` in VS Code
2. Install Jupyter extension if prompted
3. Select Python kernel
4. Click "Run All" or run cells one by one
5. See detailed guide: `VSCODE_NOTEBOOK_GUIDE.md`

**Benefits:**
- ✅ Interactive step-by-step execution
- ✅ Inline visualizations
- ✅ Progress monitoring
- ✅ Easy to experiment
- ✅ Works with or without GPU

### Option 2: Command Line

**Automated training via script:**

1. Install Dependencies
```bash
pip install -r requirements.txt
```

2. Run Training
```bash
python train_all_models.py
```

This will:
- Load and preprocess the Pride and Prejudice dataset
- Train three models (Underfit, Overfit, Best Fit)
- Generate training curves and comparison plots
- Evaluate all models on the test set
- Generate sample text using all models
- Save all results to `results/` and `models/` directories

## 📊 Model Architectures

### Small Model
- Embedding dimension: 128
- Hidden dimension: 256
- Layers: 1
- Dropout: 0.3

### Medium Model
- Embedding dimension: 256
- Hidden dimension: 512
- Layers: 2
- Dropout: 0.4

### Large Model
- Embedding dimension: 512
- Hidden dimension: 1024
- Layers: 3
- Dropout: 0.5

## 🎯 Training Configuration

- **Sequence Length**: 35 tokens
- **Batch Size**: 64
- **Learning Rate**: 0.001
- **Epochs**: 20 (with early stopping)
- **Patience**: 5 epochs
- **Gradient Clipping**: 5.0
- **Train/Val/Test Split**: 80/10/10

## 📈 Output Files

After training, you'll find:

### Models Directory (`models/`)
- `small_best.pt` - Best small model checkpoint
- `medium_best.pt` - Best medium model checkpoint
- `large_best.pt` - Best large model checkpoint
- `{model}_epoch_{n}.pt` - Periodic checkpoints

### Results Directory (`results/`)
- `small_training_curves.png` - Training/validation curves for small model
- `medium_training_curves.png` - Training/validation curves for medium model
- `large_training_curves.png` - Training/validation curves for large model
- `model_comparison_loss.png` - Comparison of validation loss across models
- `model_comparison_perplexity.png` - Comparison of perplexity across models
- `final_comparison_report.json` - Detailed metrics for all models
- `generated_samples.json` - Generated text samples

### Vocabulary Directory (`vocab/`)
- `vocab.pkl` - Saved vocabulary mappings

## 🔍 Key Features

1. **Comprehensive Data Pipeline**
   - Text tokenization and vocabulary building
   - Handling of unknown words with `<UNK>` token
   - Efficient data loading with PyTorch DataLoader

2. **Robust Training**
   - Early stopping to prevent overfitting
   - Gradient clipping for stable training
   - Regular checkpoint saving
   - Progress bars with tqdm

3. **Detailed Evaluation**
   - Loss and perplexity metrics
   - Test set evaluation
   - Model comparison reports

4. **Text Generation**
   - Temperature-controlled sampling
   - Multiple sample generation
   - Customizable sequence length

5. **Visualization**
   - Training/validation curves
   - Model comparison plots
   - Perplexity trends

## 💻 Usage Examples

### Training Individual Models

```python
from src.config import get_config
from src.model import LSTMLanguageModel
from src.train import LanguageModelTrainer

# Get configuration
config = get_config('medium')

# Create model
model = LSTMLanguageModel(
    vocab_size=vocab_size,
    embedding_dim=config['embedding_dim'],
    hidden_dim=config['hidden_dim'],
    num_layers=config['num_layers'],
    dropout=config['dropout']
)

# Train
trainer = LanguageModelTrainer(model, train_loader, val_loader, config, device)
results = trainer.train()
```

### Generating Text

```python
from src.generate import generate_text
from src.dataset import Vocabulary

# Load vocabulary
vocab = Vocabulary.load('vocab/vocab.pkl')

# Load model
checkpoint = torch.load('models/medium_best.pt')
model.load_state_dict(checkpoint['model_state_dict'])

# Generate
text = generate_text(
    model, vocab,
    start_text="it is a truth",
    max_length=50,
    temperature=0.8
)
print(text)
```

### Evaluating Models

```python
from src.evaluate import ModelEvaluator

evaluator = ModelEvaluator(model, device)
metrics = evaluator.evaluate_on_dataset(test_loader, "Test")
print(f"Test Perplexity: {metrics['perplexity']:.2f}")
```

## 📝 Expected Results

- **Training Time**: ~15-30 minutes for all three models (CPU), ~50 minutes with GPU
- **Best Model**: Large model with 1.79 test perplexity
- **Test Perplexity Results**:
  - Small: 4.58 (Underfitting)
  - Medium: 2.04 (Good fit)
  - Large: 1.79 (Best fit)
- **Generated Text**: Should maintain Jane Austen's writing style

## 🛠️ Troubleshooting

**Issue**: CUDA out of memory
- **Solution**: Reduce batch size in `config.py`

**Issue**: Slow training
- **Solution**: Use GPU if available, or reduce model size

**Issue**: Poor text generation
- **Solution**: Adjust temperature (lower = more deterministic, higher = more creative)

## 📚 Requirements

- Python 3.7+
- PyTorch 2.0.1
- NumPy 1.24.3
- Matplotlib 3.7.1
- tqdm 4.65.0
- pandas 2.0.3

## 🎓 Assignment Details

This project implements an LSTM-based language model as part of an assignment. The model learns to predict the next word in a sequence, trained on Jane Austen's "Pride and Prejudice".

Key learning objectives:
- Understanding RNN/LSTM architectures
- Implementing language modeling
- Hyperparameter tuning
- Model evaluation and comparison
- Text generation techniques

## 📄 License

This project is for educational purposes.

## 🙏 Acknowledgments

- Dataset: Project Gutenberg's "Pride and Prejudice" by Jane Austen
- Framework: PyTorch
