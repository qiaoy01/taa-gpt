import os
import json
import random
import math

class TextDataProcessor:
    def __init__(self, data_folder, config_path):
        self.data_folder = data_folder
        self.config_path = config_path

    def process_file(self, file_path, subtext_length):
        with open(file_path, encoding="utf8") as f:
            text = f.read()
            words = text.split()
            word_count = len(words)
            subtext_count = math.floor(word_count / subtext_length)
            subtext_positions = []

            for i in range(subtext_count):
                start_pos = i * subtext_length
                subtext = text[start_pos:start_pos + subtext_length]
                subtext_word_count = len(subtext.split())
                subtext_positions.append((i, start_pos, subtext_word_count))

            return subtext_length, subtext_count, subtext_positions

    def update_config(self, n_list):
        with open(self.config_path, 'r') as f:
            config = json.load(f)

        for file_name in os.listdir(self.data_folder):
            file_path = os.path.join(self.data_folder, file_name)
            for n in n_list:
                ulen, num_sub, subtext_positions = self.process_file(file_path, n)

                if file_name not in config['training_source']:
                    config['training_source'][file_name] = {}

                if str(n) not in config['training_source'][file_name]:
                    config['training_source'][file_name][str(n)] = {}

                config['training_source'][file_name][str(n)] = {
                    "ulen": ulen,
                    "num_sub": num_sub,
                    "subtext_positions": subtext_positions
                }

        with open(self.config_path, 'w') as f:
            json.dump(config, f, indent=4)

def test_subtext_word_count(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)

    file_name = random.choice(list(config['training_source'].keys()))
    n = random.choice(list(config['training_source'][file_name].keys()))
    subtext_positions = config['training_source'][file_name][n]['subtext_positions']

    file_path = os.path.join(data_folder, file_name)
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    subtext_index = random.randint(0, len(subtext_positions) - 1)
    start_position, subtext_length = subtext_positions[subtext_index][1], subtext_positions[subtext_index][2]
    subtext = content[start_position:start_position + subtext_length]
    words_count = len(subtext.split())

    print("File name:", file_name)
    print("Subtext length:", n)
    print("Subtext index:", subtext_index)
    print("Subtext start position:", start_position)
    print("Subtext actual length:", subtext_length)
    print("Subtext word count:", words_count)

    if words_count == int(n):
        print("Test passed!")
    else:
        print("Test failed!")

if __name__ == "__main__":
    data_folder = "data/openwebtext"
    config_path = "data/data_config.json"
    n_list = [4096, 8192]

    processor = TextDataProcessor(data_folder, config_path)
    processor.update_config(n_list)

    test_subtext_word_count(config_path)