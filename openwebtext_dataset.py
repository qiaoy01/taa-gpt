import os
import glob
import re
import torch
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer 

class OpenWebTextDataset(Dataset):
    def __init__(self, data_dir, tokenizer, seq_len=128):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        datapath = os.path.join(data_dir, "*")
        self.text_files = sorted(glob.glob(datapath))
    
    def preprocess_text(self, text):
        # 转换为小写
        text = text.lower()
        
        # 扩展缩写
        text = re.sub(r"won\'t", "will not", text)
        text = re.sub(r"can\'t", "cannot", text)
        # 更多的缩写可以在这里添加
        
        # 删除特殊字符
        text = re.sub(r"[^a-z0-9\s]", " ", text)
    
        return text.strip()
    
    def __len__(self):
        return len(self.text_files)
    

    def __getitem__(self, idx):

        if isinstance(idx, list):
            text_list = []
            for i in idx:
                with open(self.text_files[i], 'r', encoding='utf-8') as f:
                    text = f.read()
                text_list.append(text)
            text = ' '.join(text_list)
        else:
            with open(self.text_files[idx], 'r', encoding='utf-8') as f:
                text = f.read()
        
        # 对文本进行预处理
        text = self.preprocess_text(text)
        
        tokens = self.tokenizer.encode(self.tokenizer.tokenize(text), max_length=self.seq_len, truncation=True, padding='max_length')
        tokens = torch.tensor(tokens, dtype=torch.long)
        return {'input_ids': tokens}

# 使用预先训练的GPT-2 Tokenizer
# tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# 创建数据集实例
# data_dir = 'data/openwebtext'
# dataset = OpenWebTextDataset(data_dir, tokenizer, seq_len=128)
if __name__ == "__main__":
    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    # Create dataset instance
    data_dir = 'data/openwebtext'
    dataset = OpenWebTextDataset(data_dir, tokenizer, seq_len=128)

    if len(dataset) == 0:
        print("The dataset is empty.")
    else:
        print("The dataset has", len(dataset), "elements.")

    # Get a random sample from the dataset
    sample = dataset[torch.randint(0, len(dataset), (1,))]

    # Print the input_ids
    print(sample['input_ids'])
