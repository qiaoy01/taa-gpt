import os
import spacy

nlp = spacy.load('en_core_web_sm')
nlp.max_length = 10000000  # Increase max length to handle long texts

input_folder = 'data_folder'
output_folder = 'data'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Process each text file in data_folder
for filename in os.listdir(input_folder):
    if filename.endswith('.txt'):
        # Read in text
        with open(os.path.join(input_folder, filename), 'r', encoding='utf-8') as f:
            text = f.read()

        # Preprocess text with spaCy
        doc = nlp(text)
        tokens = [token.lemma_.lower() for token in doc if not token.is_stop and token.is_alpha]
        preprocessed_text = ' '.join(tokens)

        # Write preprocessed text to output file
        out_filename = os.path.join(output_folder, filename)
        with open(out_filename, 'w', encoding='utf-8') as f:
            f.write(preprocessed_text)

