import os
import glob
import re
import torch
from transformers import AutoTokenizer
from torch.utils.data import Dataset
import random
import json

class TaaDataset(Dataset):

    def __init__(self, data_dir, tokenizer, seq_len=128, threshold=2, max_cached_data = 10000):
        self.seq_len = seq_len
        self.text_files = sorted(glob.glob(os.path.join(data_dir, "*")))
        self.tokenizer = tokenizer
        #check size if the data
        self.threshold = threshold
        total_sequences = 0
        self.file_index = {}
        self.samples = {} #[idx] = [file_id, start_position, read_length, isChecked]
        self.data = {}
        self.max_cached_data = max_cached_data

        fileseq = 0
        for file_path in self.text_files:
            self.file_index[fileseq] = file_path
   
            # Get the file size in bytes
            file_size = os.path.getsize(file_path)
            
            num_sequences, remainder = divmod(file_size, seq_len)
            
            if remainder > self.threshold:
                num_sequences += 1
            
            for i in range(num_sequences):
                total_sequences = total_sequences + 1
                self.samples[total_sequences] = [fileseq, seq_len * i, seq_len, False] 
           
            fileseq += 1
        self.data_len = total_sequences

    def read_utf8_file_partial(file_path, start_char, num_chars):
        buffer_size = num_chars * 4
        with open(file_path, 'rb') as file:
            # 寻找开始的字符位置
            start_byte = 0
            while start_char > 0:
                byte = file.read(1)
                if not byte:
                    break
                start_byte += 1
                if (byte[0] & 0xC0) != 0x80:  # 检查是否为UTF-8的首字节
                    start_char -= 1
            
            # 从找到的位置开始读取
            result_str = ''
            while len(result_str) < num_chars:
                bytes_data = file.read(buffer_size)
                if not bytes_data:
                    break
                decoded_str = bytes_data.decode('utf-8', errors='ignore')
                result_str += decoded_str
                buffer_size = (num_chars - len(result_str)) * 4
            return result_str[:num_chars]

    def read_text_portion(self, file_path, start_position, read_length, seq_len, is_checked=False):
        adjusted_start = 0
        adjusted_end = 0

        with open(file_path, 'r', encoding='utf-8') as f:
            f.seek(start_position)
            text = f.read(read_length)

        if not is_checked:
            # Adjust the start position to the beginning of the word
            if start_position > 0:
                match = re.search(r'\b', text)
                if match:
                    adjusted_start = match.start()
                    with open(file_path, 'r', encoding='utf-8') as f:
                        f.seek(start_position + adjusted_start)
                        text = f.read(read_length)

            # Adjust the end position to the end of the word
            match = re.search(r'\b', text[read_length:])
            if match:
                adjusted_end = match.start()
                with open(file_path, 'r', encoding='utf-8') as f:
                    f.seek(start_position + adjusted_start)
                    text = f.read(read_length + adjusted_end)

            # If the total text length is greater than seq_len, discard the last word
            for ii in range(10):
                if len(text) > seq_len:
                    match = re.search(r'\b\S+\s*$', text[:seq_len])
                    if match:
                        text = text[:match.start()]
                        adjusted_end = match.start() - read_length
                else:
                    break
            
            if len(text) > seq_len:
                text = text[:seq_len]
                adjusted_end = len(text) - read_length

        adjusted_start_position = start_position + adjusted_start
        adjusted_read_length = read_length + adjusted_end
        text = text.strip()

        return text, adjusted_start_position, adjusted_read_length

    def get_token_by_index(self, idx: int):
        if idx in self.data:
            return self.data[idx]
        
        file_id, start_position, read_length, isChecked =  self.samples[idx]
        filepath = self.file_index[file_id]
        text, adjusted_start_position, adjusted_read_length = self.read_text_portion(file_path=filepath, start_position=start_position,read_length=read_length,seq_len=self.seq_len, is_checked=isChecked)
        self.samples[idx] = [file_id, adjusted_start_position, adjusted_read_length, True]
        tokenized_text = self.tokenizer.encode(text, max_length=self.seq_len, truncation=True, padding="max_length")
        token_to_return = {'input_ids': torch.tensor(tokenized_text, dtype=torch.long).view(-1, self.seq_len)}
        self.data[idx] = token_to_return
        if len(self.data) > self.max_cached_data:
            keys_without_k = [key for key in self.data.keys() if key != idx]
            key_to_remove = random.choice(keys_without_k)
            del self.data[key_to_remove]
        return token_to_return


    def __len__(self):
        return self.data_len

    def __getitem__(self, idx: int):
        return self.get_token_by_index(idx)
    
    def get_random_sample(self):
        random_idx = random.randint(0, len(self) - 1)
        return self.__getitem__(random_idx)
    

    def get_random_text_sample(self):
        # Get a random sample
        sample = self.get_random_sample()
        
        # Convert input_ids back to text
        text = self.tokenizer.decode(sample['input_ids'][0], skip_special_tokens=True)
    
        return text

if __name__=="__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

     # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    with open('config.json', 'r') as f:
        config = json.load(f)

    # Load the dataset
    data_dir = 'data/openwebtext'
    dataset = TaaDataset(data_dir=data_dir,tokenizer=tokenizer, seq_len=config['seq_len'])

    # Print the number of examples in the dataset
    print("dataset length:", len(dataset))

    # Get the first example from the dataset
    example = dataset.get_random_text_sample()
    print("random sample:", example)


    if len(dataset) == 0:
        print("The dataset is empty.")
    else:
        print("The dataset has", len(dataset), "elements.")

  