# PRIME: Programmable Response and Intelligent Machine Executor | GPT-1.5

PRIME is an advanced, lightweight language model built on the foundations of Karpathy's NanoGPT. Designed for educational exploration and practical applications, PRIME offers a streamlined yet powerful approach to natural language processing.

## Features

- **Transformer Architecture**: Simplified GPT model implementation.
- **Custom Tokenizer**: Efficient text processing.
- **Training and Inference**: Comprehensive scripts for both training new models and generating text.
- **Sampling Methods**: Supports top-k and top-p sampling for varied outputs.

## 1.5?
Yes, I have reworked the approach to the model, now it's based on Karpathy's nanoGPT. Legacy project files are located in Archive folder.

## Installation

1. **Clone the Repository**:

    ```bash
    git clone https://github.com/krll-corp/PRIME.git
    cd PRIME
    ```

2. **Set Up Virtual Environment**:

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: `venv\Scripts\activate`
    ```

3. **Install Dependencies**:

    ```bash
    pip install torch numpy transformers datasets tiktoken wandb tqdm
    ```

## Usage

### Data Preparation

Prepare your dataset by converting text into tokenized binaries:

```bash
python data/shakespeare_char/prepare.py
```

### Training

Train the model with your dataset:

```bash
python train.py config/train_shakespeare_char.py
```

### Text Generation

Generate text using a trained model:

```bash
python sample.py --out_dir=out-shakespeare-char
```

## Directory Structure

- **config/**: Contains configuration files for training.
- **data/**: Directory for storing datasets and tokenized data.
- **assets/**: Resources such as images or additional data files.
- **Archive/**: Contains archived versions of the project.
- **model.py**: Defines the transformer model architecture.
- **sample.py**: Script for generating text from the model.
- **scaling_laws.ipynb**: Jupyter notebook for exploring scaling laws in transformers.
- **train.py**: Script to train the PRIME model.
- **transformer_sizing.ipynb**: Jupyter notebook for model sizing and parameter tuning.

## Configuration

Edit the `config/train_gpt2.py` to customize training parameters:

```python
# these make the total batch size be ~0.5M
# 12 batch size * 1024 block size * 5 gradaccum * 8 GPUs = 491,520
batch_size = 12
block_size = 1024
gradient_accumulation_steps = 5 * 1 # *number GPUs you use

# this makes total number of tokens be 300B
max_iters = 600000
lr_decay_iters = 600000

# eval stuff
eval_interval = 1000
eval_iters = 200
log_interval = 10

# weight decay
weight_decay = 1e-1
```

## Requirements

- torch
- numpy
- transformers
- datasets
- tqdm

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgements

Inspired by Karpathy's NanoGPT. Thanks to the OpenAI team and  AI community for their foundational and inspirational work.
