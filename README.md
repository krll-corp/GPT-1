# PRIME: Programmable Response and Intelligent Machine Executor: GPT-1

PRIME is a small language model based on the GPT-2 architecture. This project demonstrates a simplified implementation of a language model using PyTorch and the Hugging Face Transformers library. The model is trained on a small dataset of text sequences and is capable of basic text generation tasks.

This is a first actually working GPT(Generative Ptratrained Transformer) model which is trained on a little text including two chat sequences

In this repository you can find two files calles micro-5 and micro-5.5-turbo_v4 which are the first(basic) and the most recent(advanced) implementations of the code. 

## Features

- Transformer-based language model (GPT-2 architecture)
- Custom tokenizer with special tokens based on original gpt2 tokenizer
- Training and inference scripts
- Save and load model functionality
- Text generation with custom prompts
- top-k; top-p and seed for controlling responce diversity

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/krll-corp/GPT-1.git
    cd PRIME
    ```

2. Create and activate a virtual environment (optional but recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. Install the required dependencies(not implemented yet but they are torch, trandformers and datasets):
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Training the Model

To train a new model, run the script and follow the prompt:
```bash
    python file.py
```
When asked "Train new model? (Y/N):", input Y.


### Using a Pre-trained Model

If you have a pre-trained model saved, you can load it instead of training a new one. Run the script and follow the prompt:

```bash
python file_you_need.py
```
When asked "Train new model? (Y/N):", input N.

### Text Generation
After loading the model (either newly trained or pre-trained), you can generate text by entering a sequence:

```bash
Enter the sequence (or 'quit' to stop): What are transformers?
Model: Transformer models are a type of neural network architecture...
```

# Saving/Loading Model and Tokenizer

## The model and tokenizer can be saved and loaded using the provided functions in the script:

*save_model()*: Saves the model and tokenizer to the micro/micro-5 directory.

*load_model()*: Loads the model and tokenizer from the micro/micro-5 directory.

## Script Overview

**Dataset and DataLoader:** Custom dataset and data loader for handling text sequences.

**Model Configuration:** Configuration of the GPT-2 model with a smaller architecture.

**Training Function:** Function to train the model for a specified number of epochs.

**Save and Load Functions:** Functions to save and load the model and tokenizer.

**Text Generation Function:** Function to generate text based on a given prompt.

## Requirements

- torch
- transformers


## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgements

This project uses the GPT-2 architecture and the Hugging Face Transformers library. Special thanks to OpenAI and HuggingFace for their contributions to the AI community.
