import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load the saved state dictionary
state_dict = torch.load('models/taa.pt', map_location=torch.device('cpu'))

# Instantiate the GPT2LMHeadModel and load the saved state dictionary
model = GPT2LMHeadModel.from_pretrained('gpt2', state_dict=state_dict)

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

def chatbot(input_text):
    # Tokenize input text
    input_ids = tokenizer.encode(input_text, return_tensors='pt')

    # Generate response
    response = model.generate(
        input_ids=input_ids,
        max_length=50,
        temperature=0.8,
        repetition_penalty=1.2,
        do_sample=True,
        top_k=0,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    # Decode response tokens and return as string
    return tokenizer.decode(response[0], skip_special_tokens=True)

while True:
    user_input = input("You: ")
    response = chatbot(user_input)
    print("Chatbot:", response)
