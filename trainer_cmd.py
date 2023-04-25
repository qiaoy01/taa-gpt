import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
#from taa_datasets.taa_openweb_dataset import TaaOpenWebTextDataset
#from taa_datasets.openwebtext import Openwebtext
from datasets import load_dataset
from gpt_model.gpt_decoder import GPTDecoder
import os
from transformers import AutoTokenizer
import time
import json
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GPTTrainer:
    def __init__(self, model, dataset, batch_size, learning_rate, num_epochs, pad_token_id, device):
        self.device = device
        self.model = model
        self.dataset = dataset
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)
        self.lr_scheduler = StepLR(self.optimizer, step_size=1, gamma=0.1) # decay the learning rate by a factor of 0.1 every 1000 steps
        self.learning_rates = []
        self.pad_token_id = pad_token_id


    def train(self):
        #dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, collate_fn=custom_collate_fn)
        dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
        for epoch in range(self.num_epochs):
            epoch_loss = 0.0

            for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
                input_ids = batch['input_ids'].to(self.device)
            
                inputs = input_ids
                targets = input_ids

                targets = torch.cat([targets[:, -1:], targets[:, :-1]], dim=1)

                self.optimizer.zero_grad()

                # Forward pass
                outputs = self.model(inputs)

                # Compute loss
                loss = torch.nn.functional.cross_entropy(outputs.view(-1, outputs.size(-1)), targets.view(-1), ignore_index=self.pad_token_id)

                # Backward pass
                loss.backward()
                self.optimizer.step()
                self.lr_scheduler.step()
                self.learning_rates.append(self.optimizer.param_groups[0]["lr"])

                epoch_loss += loss.item()

            print(f"Epoch {epoch+1} loss: {epoch_loss/len(dataloader)}")
        
        # Save the trained model
        if not os.path.exists('models'):
            os.makedirs('models')
        model_path = os.path.join('models', 'taa.pt')
        torch.save(self.model.state_dict(), model_path)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding=False, truncation=True, max_length=1024)

def custom_collate_fn(batch):

    input_ids = [item["input_ids"].clone().detach().long() for item in batch]
    max_length = max([len(x) for x in input_ids])

    padded_input_ids = []
    for x in input_ids:
        padded_input_ids.append(torch.cat([x, torch.tensor([tokenizer.pad_token_id] * (max_length - len(x)), dtype=torch.long)], dim=0))

    input_ids_tensor = torch.stack(padded_input_ids, dim=0)

    return {"input_ids": input_ids_tensor}

def tokenize_function(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=1024)

class TorchDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.dataset[idx]['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(self.dataset[idx]['attention_mask'], dtype=torch.long),
        }

    def __len__(self):
        return len(self.dataset)

if __name__ == '__main__':

    time0 = time.time()

    # Load training parameters from config file
    with open('config.json', 'r') as f:
        config = json.load(f)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.add_special_tokens(config['special_tokens'])

    max_seq_length = config['seq_len']
    batch_size=config['batch_size']

    # Create dataset instance
    #data_dir = 'data/openwebtext'
    #tokenized_dataset = TaaOpenWebTextDataset(data_dir=data_dir, tokenizer=tokenizer, seq_len=config['seq_len'])
    dataset = load_dataset('openwebtext', split="train[:1%]")
    #dataset = dataset.select(range(1000))
    tokenized_dataset = dataset.map(tokenize_function, batched=True, batch_size=batch_size)
    torch_dataset = TorchDataset(tokenized_dataset)

    # Create GPT decoder model instance
    gpt_decoder = GPTDecoder(
        vocab_size=tokenizer.vocab_size,
        d_model=config['d_model'],
        n_head=config['n_head'],
        d_ff=config['d_ff'],
        n_layer=config['n_layer'],
        max_seq_len=max_seq_length,
        dropout=config['dropout'],
        padding_token_id=tokenizer.pad_token_id,
        device=device
    )
    
    # Create trainer instance and start training
    trainer = GPTTrainer(
        model=gpt_decoder,
        dataset=torch_dataset,
        batch_size=batch_size,
        learning_rate=config['learning_rate'],
        num_epochs=config['num_epochs'],
        device=device,
        pad_token_id=tokenizer.pad_token_id
    )

    trainer.train()

    time1 = time.time()

    print("total training time in seconds:", time1 - time0)

    plt.plot(trainer.learning_rates)
    plt.show()

