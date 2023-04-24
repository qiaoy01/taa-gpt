import spacy

class SpacyTokenizer:
    def __init__(self, model_name='en_core_web_sm', pad_token='<pad>'):
        self.nlp = spacy.load(model_name)
        self.pad_token = pad_token
        self.vocab_size = len(self.nlp.vocab)

    def tokenize(self, text):
        return [token.text for token in self.nlp(text)]

    def encode(self, tokens, max_length=None, truncation=True, padding='max_length'):
        token_ids = [self.nlp.vocab.strings[token] for token in tokens]

        if max_length:
            if truncation:
                token_ids = token_ids[:max_length]
            if padding == 'max_length':
                token_ids += [self.pad_token_id] * (max_length - len(token_ids))

        return token_ids

    def decode(self, token_ids):
        return [self.nlp.vocab.strings[token_id] for token_id in token_ids]

    def add_special_tokens(self, tokens):
        tokens.append(self.pad_token)

    @property
    def pad_token_id(self):
        return self.nlp.vocab.strings[self.pad_token]