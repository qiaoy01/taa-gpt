import sentencepiece as spm
import json
import os,re
import logging

def clean_text(text):
    # 将不是 Unicode 编码的字符替换为 unknown_token
    text = text.encode("utf-8", errors="replace").decode("utf-8", errors="replace")

    # 将制表符、换行符和回车符替换为空格
    text = re.sub(r"[\t\n\r]", " ", text)

    # 合并连续的空格为单个空格
    text = re.sub(r" +", " ", text)

    return text

class BPE_Tokenizer:
    def __init__(self, config_file="config.json"):
        with open(config_file, "r") as f:
            config = json.load(f)
        
        self.model_dir = config["model_dir"]
        self.vocab_size = config["vocab_size"]
        self.model_type = config["model_type"]
        self.model_prefix = os.path.join(self.model_dir, config["model_prefix"])
        self.output_vocab = os.path.join(self.model_dir, config["output_vocab"])
        self.sp = None

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        # 设置 SentencePiece 的日志级别为 ERROR
        logging.getLogger("sentencepiece").setLevel(logging.ERROR)

        # 抑制 GLOG 的 INFO 级别日志
        os.environ['GLOG_minloglevel'] = '2'

    def train(self, input_files):
        # 创建一个空列表来保存临时文件名
        cleaned_files = []

        for file in input_files:
            with open(file, "r", encoding="utf-8") as f:
                raw_text = f.read()
            
            cleaned_text = clean_text(raw_text)

            cleaned_file = f"{file}_cleaned.txt"
            with open(cleaned_file, "w", encoding="utf-8") as f:
                f.write(cleaned_text)

            cleaned_files.append(cleaned_file)

        # 过滤掉空行和只包含空格的行
        filtered_files = []
        for file in cleaned_files:
            with open(file, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]
            
            if lines:
                filtered_file = f"{file}_filtered.txt"
                with open(filtered_file, "w", encoding="utf-8") as f:
                    f.write("\n".join(lines))

                filtered_files.append(filtered_file)
            else:
                print(f"{file} contains only empty lines or spaces")

        # 将过滤后的输入文件合并为一个字符串，用逗号分隔
        filtered_files_str = ",".join(filtered_files)

        # 设置训练参数
        spm_params = (
            f"--input={filtered_files_str} --model_prefix={self.model_prefix} "
            f"--vocab_size={self.vocab_size} --model_type={self.model_type}"
        )

        # 训练 SentencePiece 模型
        spm.SentencePieceTrainer.Train(spm_params)

        # 加载训练好的 SentencePiece 模型
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(f"{self.model_prefix}.model")

        # 将词汇表写入文件
        with open(self.output_vocab, "w", encoding="utf-8") as f:
            for index in range(self.sp.GetPieceSize()):
                f.write(f"{self.sp.id_to_piece(index)}\n")

        # 删除临时文件
        for file in cleaned_files + filtered_files:
            os.remove(file)


    
    def load_model(self):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(f"{self.model_prefix}.model")

    def encode(self, text):
        return self.sp.encode_as_ids(text)

    def decode(self, tokens):
        return self.sp.decode_ids(tokens)


