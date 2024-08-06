import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer, DataCollatorForLanguageModeling

# Dataset
texts = [
    "OpenAI creates AI systems.<endoftext>",
    "GPT models are transformer-based language models.<endoftext>",
    "ChatGPT is a variant of GPT.<endoftext>",
    "Transformer models have revolutionized NLP.<endoftext>",
    "Language models are useful in many applications.<endoftext>",
    "GPT-2 is a popular language model.<endoftext>",
    "What are transformers?<endoftext>Transformer models are a type of neural network architecture<endoftext>Thanks!<endoftext>You're welcome! I'm always here to assist you.<endoftext>",
    "What is the best language model?<endoftext>It depends on the task you want to perform.<endoftext>Can you provide some examples?<endoftext>Yes, sure. GPT-3, BERT, and T5 are some of the popular language models.<endoftext>",

]

# Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.add_special_tokens({'pad_token': '[PAD]', 'eos_token': '<endoftext>'})
if '[PAD]' not in tokenizer.get_vocab():
    tokenizer.add_tokens(['[PAD]'])

def tokenize_function(text):
    return tokenizer(text, truncation=True, max_length=64, padding="max_length", return_tensors="pt")

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer):
        self.texts = texts
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encodings = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=64,
            padding="max_length",
            return_tensors="pt"
        )
        return encodings["input_ids"].squeeze(), encodings["attention_mask"].squeeze()


dataset = TextDataset(texts, tokenizer)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Config
config = GPT2Config(
    vocab_size=len(tokenizer),
    n_positions=64,
    n_ctx=64,
    n_embd=128,
    n_layer=4,
    n_head=4
)

model = GPT2LMHeadModel(config)
optimizer = optim.AdamW(model.parameters(), lr=3e-4)
loss_fn = nn.CrossEntropyLoss()

# Device
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def train():
    epochs = 300
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for input_ids, attention_mask in dataloader:
            input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")


# Save model and tokenizer
def save_model():
    model.save_pretrained("micro/micro-5")
    tokenizer.save_pretrained("micro/micro-5")


# Load if needed
def load_model():
    model = GPT2LMHeadModel.from_pretrained("micro/micro-5")
    tokenizer = GPT2Tokenizer.from_pretrained("micro/micro-5")
    model.to(device)
    return model, tokenizer

def generate_text(model, tokenizer, initial_text, max_length=64):
    model.eval()
    print(start, end="")
    with torch.no_grad():
        input_ids = tokenizer.encode(initial_text, return_tensors='pt').to(device)
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=device)  # Initial attention mask

        generated_text = initial_text
        for _ in range(max_length):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = outputs.logits

            # Get the predicted index
            predicted_index = torch.argmax(predictions[0, -1, :]).unsqueeze(0)
            predicted_token_id = predicted_index.unsqueeze(0)

            predicted_token = tokenizer.decode(predicted_token_id[0].tolist())
            print(predicted_token, end="")
            generated_text += predicted_token

            input_ids = torch.cat((input_ids, predicted_token_id), dim=1)
            attention_mask = torch.cat((attention_mask, torch.ones((1, 1), dtype=torch.long, device=device)), dim=1)
            if predicted_token_id[0].item() == tokenizer.eos_token_id:
                break

        return ""



if __name__ == "__main__":
    do_train = input("Train new model? (Y/N): ")
    if do_train.lower() == "y":
        train()
        save_model()
    else:
        model, tokenizer = load_model()

    while True:
        start = input("Enter the sequence (or 'quit' to stop): ")
        if start.lower() in ["quit", "exit", "stop"]:
            break
        else:
            print("Model: ", end="")
            print(generate_text(model, tokenizer, start, 50))