import torch
import random
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer, get_linear_schedule_with_warmup
from datasets import load_dataset

# Load Blended Skill Talk dataset
dataset_bst = load_dataset("blended_skill_talk")

conversation_texts = []

# Extract the conversation content from Blended Skill Talk
for dialog in dataset_bst["train"]:
    personas = " <endoftext> ".join(dialog["personas"])
    previous_utterances = " <endoftext> ".join(dialog["previous_utterance"])
    additional_context = dialog.get("additional_context", "")
    free_messages = " <endoftext> ".join(dialog["free_messages"])
    guided_messages = " <endoftext> ".join(dialog["guided_messages"])
    conversation = f"{personas} <endoftext> {previous_utterances} <endoftext> {additional_context} <endoftext> {free_messages} <endoftext> {guided_messages} <endoftext>"
    conversation_texts.append(conversation)

# Stream new version of C4 dataset
streamed_c4 = load_dataset("allenai/c4", "en", split='train', streaming=True, trust_remote_code=True)

# Function to process streamed data
def process_streamed_data(streamed_data, num_samples=1000):
    count = 0
    for example in streamed_data:
        conversation_texts.append(example['text'])
        count += 1
        if count >= num_samples:
            break

# Process 1000 samples for the first run
process_streamed_data(streamed_c4, num_samples=1000)

# Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.add_special_tokens({'eos_token': '<endoftext>', 'pad_token': '[PAD]'})
if '[PAD]' not in tokenizer.get_vocab():
    tokenizer.add_tokens(['[PAD]'])
    tokenizer.pad_token = '[PAD]'

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
            max_length=256,
            padding="max_length",
            return_tensors="pt"
        )
        return encodings["input_ids"].squeeze(), encodings["attention_mask"].squeeze()

dataset = TextDataset(conversation_texts, tokenizer)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Config
config = GPT2Config(
    vocab_size=len(tokenizer),
    n_positions=256,
    n_ctx=256,
    n_embd=128,
    n_layer=8,
    n_head=8
)

model = GPT2LMHeadModel(config)
optimizer = optim.AdamW(model.parameters(), lr=5e-5)
loss_fn = nn.CrossEntropyLoss()

# Device
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Scheduler
total_steps = len(dataloader) * 100
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

def train():
    epochs = 100
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
            scheduler.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

# Save model and tokenizer
def save_model():
    model.save_pretrained("micro/micro-5.5-turbo_v4")
    tokenizer.save_pretrained("micro/micro-5.5-turbo_v4")

# Load if needed
def load_model():
    model = GPT2LMHeadModel.from_pretrained("micro/micro-5.5-turbo_v4")
    tokenizer = GPT2Tokenizer.from_pretrained("micro/micro-5.5-turbo_v4")
    model.to(device)
    return model, tokenizer

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def generate_text(model, tokenizer, initial_text, max_length=256, top_k=0, top_p=0.0, debug=False):
    model.eval()
    with torch.no_grad():
        input_ids = tokenizer.encode(initial_text, return_tensors='pt').to(device)
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=device)

        generated_text = initial_text
        for _ in range(max_length):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = outputs.logits[:, -1, :]

            if top_k > 0:
                top_k_values, top_k_indices = torch.topk(predictions, top_k)
                top_k_probabilities = F.softmax(top_k_values, dim=-1)
                top_k_index = torch.multinomial(top_k_probabilities, 1)
                predicted_index = top_k_indices[0, top_k_index].squeeze()
            elif top_p > 0.0:
                sorted_logits, sorted_indices = torch.sort(predictions, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                predictions[:, indices_to_remove] = -float('Inf')
                predicted_index = torch.multinomial(F.softmax(predictions, dim=-1), 1).squeeze()
            else:
                predicted_index = torch.argmax(predictions, dim=-1).squeeze()

            predicted_token_id = predicted_index.unsqueeze(0)
            predicted_token = tokenizer.decode(predicted_token_id.tolist())

            if debug and top_k > 0:
                print(f"\nTop-{top_k} tokens and probabilities:")
                for i in range(top_k):
                    token_id = top_k_indices[0, i].item()
                    token = tokenizer.decode([token_id])
                    prob = top_k_probabilities[0, i].item()
                    print(f"{token} ({prob:.4f})")

            print(predicted_token, end="")
            generated_text += predicted_token

            input_ids = torch.cat((input_ids, predicted_token_id.unsqueeze(0)), dim=1)
            attention_mask = torch.cat((attention_mask, torch.ones((1, 1), dtype=torch.long, device=device)), dim=1)
            if predicted_token_id.item() == tokenizer.eos_token_id:
                break
            if input_ids.shape[1] > model.config.n_ctx:
                input_ids = input_ids[:, -model.config.n_ctx:]
                attention_mask = attention_mask[:, -model.config.n_ctx:]
        if debug:
            print(generated_text)
        return None

if __name__ == "__main__":
    do_train = input("Train new model? (Y/N): ").lower()
    debug_mode = input("Debug mode? (D/N): ").lower() == "d"

    randseed = False

    if do_train == "y":
        train()
        save_model()
    model, tokenizer = load_model()
    top_k = int(input("Enter top-k (0 for no top-k): "))
    top_p = float(input("Enter top-p (0 for no top-p): "))
    seed = int(input("Enter seed (0 for no seed): "))

    randseed = seed == 0
    if not randseed:
        set_seed(seed)

    while True:
        if randseed:
            set_seed(random.randint(1, 10000))
        start = input("Enter the sequence (or 'quit' to stop): ").strip()
        if start.lower() in ["quit", "exit", "stop"]:
            break
        else:
            print("Model: ", end="")
            print(start, end="")
            generated_text = generate_text(model, tokenizer, start, max_length=256, top_k=top_k, top_p=top_p, debug=debug_mode)
            print(generated_text)