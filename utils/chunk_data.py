import torch
from transformers import BertTokenizer
from datasets import load_dataset
import os
import tqdm

def tokenize_and_chunk_data(data, tokenizer):
    tokenized_data = tokenizer.encode_plus(data, padding=True, truncation=True, return_tensors='pt')
    input_ids = tokenized_data['input_ids']

    num_chunks = (input_ids.shape[1] + 2048) // 2049
    padded_length = num_chunks * 2049
    padded_input_ids = torch.zeros((input_ids.shape[0], padded_length), dtype=torch.long)
    padded_input_ids[:, :input_ids.shape[1]] = input_ids

    chunks = []
    for i in range(num_chunks):
        chunk = padded_input_ids[:, i*2049:(i+1)*2049]
        chunks.append(chunk)

    return chunks

def convert_and_save_data(dataset, output_path, tokenizer):
    # Initialize the binary file
    with open(output_path, 'wb') as f:
        pass

    # Iterate through the dataset with a progress bar
    for idx, data in enumerate(tqdm.tqdm(dataset, desc="Processing dataset")):
        # Tokenize and chunk the data
        chunks = tokenize_and_chunk_data(data['text'], tokenizer)

        # Save the chunks to a binary file on disk
        with open(output_path, 'ab') as f:
            for chunk in chunks:
                # Write the first 2048 tokens and the 1-2049 tokens as separate chunks
                chunk_data = chunk[:, :2048].cpu().numpy().tobytes()
                f.write(chunk_data)
                chunk_data = chunk[:, 1:2049].cpu().numpy().tobytes()
                f.write(chunk_data)

        # Print progress
        if (idx + 1) % 1000 == 0:
            print(f"Processed {idx + 1} samples")

# Load the OpenWebText dataset
openwebtext = load_dataset('openwebtext')

# Set the tokenizer name and output path
tokenizer_name = 'bert-base-uncased'
output_path = 'output.bin'

# Load the tokenizer
tokenizer = BertTokenizer.from_pretrained(tokenizer_name)

# Convert and save the dataset
convert_and_save_data(openwebtext['train'], output_path, tokenizer)
