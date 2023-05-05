"""
Trains a GPT to add n-digit numbers.
"""

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from core.model import GPTDecoder
from trainer.default_trainer import Trainer
import json
from utils.web_loss_curve import submit_loss_data

def tokenize_function(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=2049)

class CasualDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, idx):
        sample = self.dataset[idx]['input_ids']
        x = torch.tensor(sample[:-1], dtype=torch.long)
        y = torch.tensor(sample[1:], dtype=torch.long)
        return x, y

    def __len__(self):
        return len(self.dataset)
    
if __name__ == '__main__':

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  
    with open('config.json','r') as f:
        config = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token_id = tokenizer.eos_token_id
    textdataset = load_dataset('openwebtext', split="train[:1%]")
    textdataset = textdataset.select(range(1000))
    tokenized_dataset = textdataset.map(tokenize_function, batched=True)
    train_dataset = CasualDataset(tokenized_dataset)
    vocab_size = tokenizer.vocab_size
    max_seq_len = config['seq_len']


    model = GPTDecoder(vocab_size=vocab_size, 
                       d_model=config['d_model'], 
                       num_heads=config['n_head'], 
                       d_ff=config['d_ff'],
                       num_layers=config['n_layer'], 
                       max_seq_len=max_seq_len, 
                       causal_attention=True, 
                       dropout_rate=config['dropout']).to(device)
    

    trainer = Trainer(model=model,train_dataset=train_dataset, config_file='config.json')
    trainer.run()
    