import os
import glob
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer

class TextDataset(Dataset):
    def __init__(self, data_folder):
        self.tokenizer =  GPT2Tokenizer.from_pretrained('gpt2')
        # Set the padding token to be the same as the end-of-sequence (EOS) token
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.data_folder = data_folder
        self.file_paths = glob.glob(os.path.join(data_folder, '*.txt'))
        self.text_data = []

        for file_path in self.file_paths:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                self.text_data.append(text)

    def __len__(self):
        return len(self.text_data)

    def __getitem__(self, index):
        text = self.text_data[index]
        tokenized_text = self.tokenizer(text, return_tensors='pt', truncation=True, padding='max_length', max_length=1024)
        return tokenized_text

if __name__ == '__main__':
    data_folder = 'data_folder'
    dataset = TextDataset(data_folder)

    # Example: accessing an item from the dataset
    sample = dataset[0]
    print(sample)

