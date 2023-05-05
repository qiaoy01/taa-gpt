"""
Trains a GPT to add n-digit numbers.
"""

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

from core.model import GPTDecoder
from trainer.default_trainer import Trainer
import json
from utils.web_loss_curve import submit_loss_data



'''
AddtionDataset from mingpt
'''
class AdditionDataset(Dataset):


    def __init__(self, split):
        self.split = split # train/test

        # split up all addition problems into either training data or test data
        ndigit = 2
        assert ndigit <= 3, "the lines below would be very memory inefficient, in future maybe refactor to support"
        num = (10**ndigit)**2 # total number of possible addition problems with ndigit numbers
        rng = torch.Generator()
        rng.manual_seed(1337)
        perm = torch.randperm(num, generator=rng)
        num_test = min(int(num*0.2), 500) # 20% of the whole dataset, or only up to 500
        self.ixes = perm[:num_test] if split == 'test' else perm[num_test:]

    def get_vocab_size(self):
        return 10 # 0-9

    def get_block_size(self):
        return 6 # len('5050100') = 7 -1 = 6

    def __len__(self):
        return self.ixes.nelement()

    def __getitem__(self, idx):
        ndigit = 2
        # given a problem index idx, first recover the associated a + b
        idx = self.ixes[idx].item()
        nd = 10**ndigit
        a = idx // nd
        b = idx %  nd
        # calculate the "label" of the addition problem a + b
        c = a + b
        # encode the digits of a, b, c into strings
        astr = f'%0{ndigit}d' % a
        bstr = f'%0{ndigit}d' % b
        cstr = (f'%0{ndigit+1}d' % c)[::-1] # reverse c to make addition easier
        render = astr + bstr + cstr
        dix = [int(s) for s in render] # convert each character to its token index
        # x will be input to GPT and y will be the associated expected outputs
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long) # predict the next token in the sequence
        y[:ndigit*2-1] = -1 # we will only train in the output locations. -1 will mask loss to zero
        return x, y

if __name__ == '__main__':

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  

    # construct train and test datasets
    train_dataset = AdditionDataset(split='train')
    test_dataset  = AdditionDataset(split='test')

    # construct the model
    vocab_size = train_dataset.get_vocab_size()
    max_seq_len = train_dataset.get_block_size()

    with open("config.json", 'r') as f:
        config = json.load(f)
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
    