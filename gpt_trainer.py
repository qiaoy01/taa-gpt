import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
#from text_dataset import TextDataset
#from openwebtext_dataset import OpenWebTextDataset
#from my_openweb_dataset import MyOpenWebTextDataset
from taa_dataset import TaaDataset
from gpt_decoder import GPTDecoder
import os
from transformers import AutoTokenizer
import json
#from spacy_tokenizer import SpacyTokenizer
#from custom_tokenizer import CustomTokenizer
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GPTTrainer:
    def __init__(self, model, dataset, batch_size, learning_rate, num_epochs, device):
        self.device = device
        self.model = model
        self.dataset = dataset
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)
        self.lr_scheduler = StepLR(self.optimizer, step_size=1, gamma=0.1) # decay the learning rate by a factor of 0.1 every 1000 steps
        self.learning_rates = []

    def train(self):
        dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.num_epochs):
            epoch_loss = 0.0

            for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
                inputs = batch['input_ids'].to(self.device)
                targets = batch['input_ids'].to(self.device)

                # 将目标向右移动一个位置
                targets = torch.cat([targets[:, -1:], targets[:, :-1]], dim=1)

                self.optimizer.zero_grad()

                # Forward pass
                outputs = self.model(inputs)

                # Compute loss
                loss = torch.nn.functional.cross_entropy(outputs.view(-1, outputs.size(-1)), targets.view(-1))

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

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load training parameters from config file
    with open('config.json', 'r') as f:
        config = json.load(f)

    # Load tokenizer
    #tokenizer = CustomTokenizer(vocab_path="vocab/vocabulary.txt")
    #tokenizer.add_special_tokens(['<pad>'])
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    # Create dataset instance
    data_dir = 'data/openwebtext'
    dataset = TaaDataset(data_dir=data_dir, tokenizer=tokenizer, seq_len=config['seq_len'], threshold=2, max_cached_data = 10 )
    
    # Create GPT decoder model instance
    gpt_decoder = GPTDecoder(
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
    
    # Create trainer instance and start training
    trainer = GPTTrainer(
        model=gpt_decoder,
        dataset=dataset,
        batch_size=config['batch_size'],
        learning_rate=config['learning_rate'],
        num_epochs=config['num_epochs'],
        device=device
    )
    trainer.train()

    plt.plot(trainer.learning_rates)
    plt.show()
