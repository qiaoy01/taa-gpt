import os
from gpt_model.bpe_tokenizer import BPE_Tokenizer
import json

import tempfile

def remove_non_utf8_chars(input_file, unknown_token="<unk>"):
    with tempfile.NamedTemporaryFile(mode="w", delete=False, encoding='utf-8') as temp_file:
        with open(input_file, 'r', encoding='utf-8', errors='replace') as file:
            for line in file:
                cleaned_line = line.replace("\ufffd", unknown_token)
                temp_file.write(cleaned_line)
    
    return temp_file.name


def main():
    # 创建 BPE_Tokenizer 实例
    tokenizer = BPE_Tokenizer()

    # 从配置文件中获取训练数据文件夹路径
    with open("config.json", "r") as f:
        config = json.load(f)
    training_data_dir = config["training_data_dir"]

    # 遍历文件夹，获取所有训练文件的路径
    input_files = [
        os.path.join(training_data_dir, file)
        for file in os.listdir(training_data_dir)
        if os.path.isfile(os.path.join(training_data_dir, file))
    ]

    # 移除训练文件中的非 UTF-8 编码字符
    filtered_files = [remove_non_utf8_chars(file_path) for file_path in input_files]

    # 使用训练文件列表训练 BPE 模型
    tokenizer.train(filtered_files)

    # 对文本进行编码和解码
    sample_text = "This is a sample text for testing the BPE tokenizer."
    encoded_text = tokenizer.encode(sample_text)
    print(f"Encoded text: {encoded_text}")

    decoded_text = tokenizer.decode(encoded_text)
    print(f"Decoded text: {decoded_text}")

    # 删除临时文件
    for temp_file in filtered_files:
        os.remove(temp_file)

if __name__ == "__main__":
    main()
