import time
import torch.nn.functional as F
from torch.utils.data import RandomSampler
from torch.nn.utils import clip_grad_norm_
from utils.web_loss_curve import submit_loss_data

import torch
from torch.utils.data.dataloader import DataLoader
import json
import os

class Trainer:

    def __init__(self,  model, train_dataset, config_file='config.json'):
        with open(config_file, 'r') as f:
            self.config = json.load(f)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.betas = (self.config['beta1'], self.config['beta2'])
        self.learing_rate = self.config['learning_rate']
        self.model = model.to(self.device)
        
        if self.config['optimizer']=='AdamW':
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config['learning_rate'], betas=self.betas)
        elif self.config['optimizer'] == 'SDG':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.config['learning_rate'])
        
        self.train_dataset = train_dataset

        self.iter_num = 0
        self.iter_time = 0.0
        self.iter_dt = 0.0
        self.max_iters = self.config['num_epochs']
        self.show_loss_curve = self.config['show_loss_curve']
        self.model_path = self.config['model_dir']
        self.model_name = self.config['model_name']
        self.checkpoint = self.config['checkpoint']

        self.checkpoint_path = os.path.join(self.model_path, self.model_name)

    def run(self):
        model = self.model
        
        train_loader = DataLoader(
            self.train_dataset,
            sampler=RandomSampler(self.train_dataset, replacement=True, num_samples=None),
            shuffle=False,
            pin_memory=True,
            batch_size=self.config['batch_size'],
            num_workers=self.config['num_workers'],
        )

        model.train()
        self.iter_num = 0
        self.iter_time = time.time()
        data_iter = iter(train_loader)
        while True:

            try:
                batch = next(data_iter)
            except StopIteration:
                print("loader error")
                data_iter = iter(train_loader)
                batch = next(data_iter)
            batch = [t.to(self.device) for t in batch]
            x, y = batch

            outputs = model(x)
            self.loss = F.cross_entropy(outputs.view(-1, outputs.shape[-1]), y.view(-1), ignore_index=-1)
            print(f"iter_dt {self.iter_dt * 1000:.2f}ms; iter {self.iter_num}: train loss {self.loss.item():.5f}")
            
            if self.show_loss_curve:
                submit_loss_data(iteration=self.iter_num, loss=self.loss.item())
            
            model.zero_grad(set_to_none=True)
            self.loss.backward()
            clip_grad_norm_(model.parameters(), self.config['grad_norm_clip'])
            self.optimizer.step()

            self.iter_num += 1
            tnow = time.time()
            self.iter_dt = tnow - self.iter_time
            self.iter_time = tnow

            if self.iter_num % self.checkpoint == 0:
                torch.save(model.state_dict(), self.checkpoint_path)

            if self.max_iters is not None and self.iter_num >= self.max_iters:
                break