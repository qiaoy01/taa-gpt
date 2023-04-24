import os
import glob
import re
import datasets
import torch
from transformers import AutoTokenizer

_CITATION = """\
@misc{Gokaslan2019OpenWeb,
  title={OpenWebText Corpus},
  author={Aaron Gokaslan*, Vanya Cohen*, Ellie Pavlick, Stefanie Tellex},
  howpublished{\\url{http://Skylion007.github.io/OpenWebTextCorpus}},
  year={2019}
}
"""

_DESCRIPTION = """\
An open-source replication of the WebText dataset from OpenAI.
"""

class MyOpenWebTextDataset(datasets.Dataset):
    """
    The Open WebText dataset.

    Attributes:
        data_dir (str): The directory containing the dataset files.
        seq_len (int): The maximum length for tokenization.
        text_files (list): A list of dataset file paths.
        tokenizer (datasets.Tokenizer): The tokenizer to preprocess the text.
    """
    def __init__(self, data_dir, tokenizer, seq_len=128):
        self.seq_len = seq_len
        self.text_files = sorted(glob.glob(os.path.join(data_dir, "*")))
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.text_files)

    def __getitem__(self, idx: int):
        
        print(idx)
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
        
        #print(text)

        # Tokenize and truncate the text
        tokenized_text = self.tokenizer.encode(text, max_length=self.seq_len, truncation=True, padding="max_length")

        return {'input_ids': torch.tensor(tokenized_text, dtype=torch.long).view(-1, self.seq_len)}
        
    def get_examples(self):
        """Yield examples from the dataset."""
        for txt_file in self.text_files:
            with open(txt_file, 'r', encoding='utf-8') as f:
                text = re.sub("\n\n\n+", "\n\n", f.read().strip())
                yield text

if __name__=="__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

     # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    # Load the dataset
    data_dir = 'data/openwebtext'
    dataset = MyOpenWebTextDataset(data_dir=data_dir,tokenizer=tokenizer)

    # Print the number of examples in the dataset
    print(len(dataset))

    # Get the first example from the dataset
    example = next(iter(dataset.get_examples()))
    print(example)


    if len(dataset) == 0:
        print("The dataset is empty.")
    else:
        print("The dataset has", len(dataset), "elements.")

    # Get a random sample from the dataset
    sample = dataset[torch.randint(0, len(dataset), (1,))]

    # Print the input_ids
    print(sample['input_ids'])
