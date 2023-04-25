from gpt_model.gpt_decoder import GPTDecoder
from transformers import AutoTokenizer
import torch
import json

def test_trained_gpt_decoder():
    # Load your pre-trained model
    model_path = "models/taa.pt"
    state_dict = torch.load(model_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load training parameters from config file
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    # Initialize GPTDecoder model with the same parameters as your pre-trained model
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    #model = GPT2LMHeadModel.from_pretrained('gpt2-large').to(device)
    tokenizer.add_special_tokens(config['special_tokens'])
    
    model = GPTDecoder(
        vocab_size=tokenizer.vocab_size,
        d_model=config['d_model'],
        n_head=config['n_head'],
        d_ff=config['d_ff'],
        n_layer=config['n_layer'],
        max_seq_len=config['seq_len'],
        dropout=config['dropout'],
        padding_token_id=tokenizer.pad_token_id,
        device=device
    )
    model.load_state_dict(state_dict)
    
    # Input sentence
    input_text = "Hello, I am Chatty Fish, how are you?"

    # Tokenize input text
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
    print("Input id:", input_ids)
    input_tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    input_text = tokenizer.decode(input_ids[0], clean_up_tokenization_spaces=True)
    print("Input Tokens:", input_tokens)
    print("Input Text:", input_text)
    # create attention mask
    #attention_mask = torch.ones_like(input_ids)
    #attention_mask[input_ids == tokenizer.pad_token_id] = 0

    # Generate text using your model
    output = model.generate(input_ids).to(device)
    print("Output id:", output[0][0])
    output_tokens = tokenizer.convert_ids_to_tokens(output[0][0])
    print("Output tokens:", output_tokens)

    # Decode output tokens
    generated_text = tokenizer.decode(output[0][0], skip_special_tokens=True, clean_up_tokenization_spaces=True)

    # Test if the generated_text is not empty
    assert generated_text, "The generated text is empty"

    print(f"Generated text: {generated_text}")
    print("Custom GPT model test passed!")

if __name__ == "__main__":
    test_trained_gpt_decoder()



