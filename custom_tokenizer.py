import spacy
import torch
import os

class CustomTokenizer:
    def __init__(self, vocab_path):
        with open(vocab_path, 'r', encoding='utf-8') as f:
            self.vocab = [line.strip() for line in f.readlines()]

        self.nlp = spacy.load("en_core_web_lg", disable=["parser", "ner", "tagger", "attribute_ruler"])
        self.unk_token = '<unk>'
        self.pad_token = '<pad>'
        self.vocab_size = len(self.vocab)

        # 创建token到ID的映射
        self.token2id = {token: i for i, token in enumerate(self.vocab)}
        self.id2token = {i: token for token, i in self.token2id.items()}
        self.pad_token_id = self.token2id[self.pad_token]

    def tokenize(self, text):
        doc = self.nlp(text)
        tokens = [token.text for token in doc]
        return tokens

    def encode(self, text):
        tokens = self.tokenize(text)
        token_ids = [self.token2id.get(token, self.token2id[self.unk_token]) for token in tokens]
        return token_ids

    def decode(self, token_ids):
        tokens = [self.id2token[token_id] for token_id in token_ids]
        text = " ".join(tokens)
        return text

    def add_special_tokens(self, special_tokens):
        for token in special_tokens:
            if token not in self.token2id:
                self.token2id[token] = len(self.vocab)
                self.id2token[len(self.vocab)] = token
                self.vocab.append(token)
                self.vocab_size += 1

# 示例
if __name__ == '__main__':
    vocab_path = os.path.join('vocab', 'vocabulary.txt')
    tokenizer = CustomTokenizer(vocab_path)
    text = "I wrote a long theme"
    token_ids = tokenizer.encode(text)
    decoded_text = tokenizer.decode(token_ids)
    print(f"Encoded: {token_ids}")
    print(f"Decoded: {decoded_text}")

