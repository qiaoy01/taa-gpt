import spacy
from spacy.tokens import Doc
from collections import Counter
import os
import json
import glob
import re

def clean_token(token):
    if token.is_alpha:
        return token.text.lower()
    else:
        return None

def build_vocabulary(data_dir, vocab_output_path, min_freq=5):
    nlp = spacy.load("en_core_web_lg", disable=["parser", "ner"])
    # Add 'tagger' and 'attribute_ruler' components to the pipeline
    if "tagger" not in nlp.pipe_names:
        nlp.add_pipe("tagger")
    
    if "attribute_ruler" not in nlp.pipe_names:
        nlp.add_pipe("attribute_ruler")

    nlp.max_length = 50000000  # Increase the maximum length limit
    counter = Counter()

    text_files = sorted(glob.glob(os.path.join(data_dir, "*.txt")))

    for file in text_files:
        with open(file, 'r', encoding='utf-8') as f:
            text = f.read()
        doc = nlp(text)
        cleaned_tokens = [clean_token(token) for token in doc if clean_token(token) is not None]
        counter.update(cleaned_tokens)

    os.makedirs(os.path.dirname(vocab_output_path), exist_ok=True)
    with open(vocab_output_path, 'w', encoding='utf-8') as f:
        f.write('<unk>\n')
        f.write('<pad>\n')
        for token, freq in counter.most_common():
            if freq >= min_freq:
                f.write(f"{token}\n")

if __name__ == '__main__':
    data_dir = 'data/openwebtext'
    vocab_output_path = os.path.join('vocab', 'vocabulary.txt')
    build_vocabulary(data_dir, vocab_output_path)


