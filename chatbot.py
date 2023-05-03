import torch
from transformers import AutoTokenizer
import json
from core.model import GPTDecoder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Load the saved state dictionary
state_dict = torch.load('models/taa.pt', map_location=torch.device('cpu'))

with open('config.json', 'r') as f:
    config = json.load(f)

tokenizer = AutoTokenizer.from_pretrained('gpt2')
tokenizer.add_special_tokens(config['special_tokens'])
model = GPTDecoder(
        vocab_size=tokenizer.vocab_size,
        d_model=config['d_model'],
        num_heads=config['n_head'],
        d_ff=config['d_ff'],
        num_layers=config['n_layer'],
        max_seq_len=config['seq_len'],
        dropout_rate=config['dropout'],
        causal_attention=True
    ).to(device)

def chatbot(input_text):
    # Tokenize input text
    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)

    # Generate response
    response = model.generate(
        input_ids=input_ids,
        max_length=50,
        temperature=0.8,
        top_k=0,
    )

    # Decode response tokens and return as string
    return tokenizer.decode(response[0][0], skip_special_tokens=True)

while True:
    user_input = input("You: ")
    response = chatbot(user_input)
    print("Chatbot:", response)
