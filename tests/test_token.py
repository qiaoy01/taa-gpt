from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('gpt2')
print(tokenizer.unk_token)
print(tokenizer.unk_token_id)
print(tokenizer.pad_token)
print(tokenizer.pad_token_id)
print(tokenizer.eos_token)
print(tokenizer.eos_token_id)

