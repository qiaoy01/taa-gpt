import torch
from transformers import BertTokenizer
from datasets import load_dataset
import os

def tokenize_and_chunk_data(data, tokenizer):
    tokenized_data = tokenizer.batch_encode_plus(data, padding=True, truncation=True, return_tensors='pt')
    input_ids = tokenized_data['input_ids']

    num_chunks = (input_ids.shape[1] + 2047) // 2048
    padded_length = num_chunks * 2048
    padded_input_ids = torch.zeros((input_ids.shape[0], padded_length), dtype=torch.long)
    padded_input_ids[:, :input_ids.shape[1]] = input_ids

    chunks = []
    for i in range(num_chunks):
        chunk = padded_input_ids[:, i*2048:(i+1)*2048]
        chunks.append(chunk)

    return chunks

def convert_and_save_data(data_dir, output_path, tokenizer_name):
    # Load the data from the dataset
    with open(os.path.join(data_dir, 'data.txt'), 'r') as f:
        data = [line.strip() for line in f.readlines()]

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Tokenize and chunk the data
    chunks = tokenize_and_chunk_data(data, tokenizer)

    # Save the data to a binary file on disk
    with open(output_path, 'wb') as f:
        for chunk in chunks:
            chunk_data = chunk.cpu().numpy().tobytes()
            f.write(chunk_data)
