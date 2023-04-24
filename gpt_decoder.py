import torch
import torch.nn as nn
import math
from gpt_decoder_layer import GPTDecoderLayer
from transformers import GPT2Tokenizer
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

class GPTDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_head, d_ff, n_layer, max_seq_len, padding_token_id, device, dropout=0.1):
        super(GPTDecoder, self).__init__()

        self.device = device
        self.embedding = nn.Embedding(vocab_size, d_model, device=device)
        self.pos_encoding = self.create_positional_encoding(d_model, max_seq_len)

        self.layers = nn.ModuleList([GPTDecoderLayer(d_model=d_model, n_head=n_head, d_ff=d_ff, dropout=dropout, device=device) for _ in range(n_layer)])

        self.fc = nn.Linear(d_model, vocab_size, bias=False, device=device)

        self.padding_token_id = padding_token_id


    def forward(self, input_tokens):
        x = input_tokens.to(self.device)
        e = self.embedding(x)
        p = self.pos_encoding[:x.size(1), :]
        input_embedded = e + p
        combined_mask = self.create_combined_mask(x)

        for layer in self.layers:
            input_embedded = layer(input_embedded, mask=combined_mask)

        output = self.fc(input_embedded)

        return output

    def create_positional_encoding(self, d_model, max_seq_len):
        pos_encoding = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pos_encoding[pos, i] = math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pos_encoding[pos, i+1] = math.cos(pos / (10000 ** ((2 * (i+1)) / d_model)))
        pos_encoding = pos_encoding.float()
        return pos_encoding.to(self.device)

    def create_causal_mask(self, input_tokens):
        device = input_tokens.device

        # Create the causal mask
        causal_mask = torch.tril(torch.ones(input_tokens.shape[1], input_tokens.shape[1])).bool().to(device)

        return causal_mask

    def create_combined_mask(self, input_tokens):

        # Create the causal mask
        causal_mask = self.create_causal_mask(input_tokens=input_tokens)

        # Create the padding mask
        padding_mask = (input_tokens == self.padding_token_id).unsqueeze(1).unsqueeze(2).to(self.device)

        # Combine the masks
        combined_mask = causal_mask | padding_mask

        return combined_mask
    
 
    def generate(self, input_ids, max_length=50, num_return_sequences=1, temperature=1.0, top_k=0):
        generated_sequences = []
        for _ in range(num_return_sequences):
            sequence = input_ids
            for _ in range(max_length - len(input_ids[0])):
                with torch.no_grad():
                    # Get logits for the next token
                    logits = self.forward(sequence)[0][:, -1] / temperature

                    # Apply top-k sampling
                    top_k = min(top_k, logits.size(-1))
                    if top_k > 0:
                        top_k_values, top_k_indices = torch.topk(logits, top_k)
                        next_token = torch.multinomial(F.softmax(top_k_values, dim=-1), 1)
                        next_token = top_k_indices.gather(-1, next_token)
                    else:
                        next_token = torch.multinomial(F.softmax(logits, dim=-1), 1)

                # Add the next token to the sequence
                next_token = next_token.unsqueeze(-1)
                sequence = torch.cat((sequence, next_token), dim=-1)

                # Limit the generated sequence length to the size of the positional encoding matrix
                if sequence.size(1) >= self.pos_encoding.size(0):
                    break

            generated_sequences.append(sequence[:, input_ids.shape[1]:].squeeze(-1))
        return torch.stack(generated_sequences, dim=0)
    
    def visualize_positional_encoding(self):
        pos_encoding = self.pos_encoding.detach().cpu().numpy()
        plt.figure(figsize=(15, 5))
        plt.pcolormesh(pos_encoding, cmap='viridis')
        plt.xlabel('Embedding Dimension')
        plt.ylabel('Position')
        plt.colorbar()
        plt.title('Positional Encoding')
        plt.show()
    
    def plot_causal_mask(self, causal_mask):
        plt.figure(figsize=(8, 8))
        plt.imshow(causal_mask.cpu().numpy(), cmap='viridis')
        plt.xlabel("Query Positions")
        plt.ylabel("Key Positions")
        plt.colorbar()
        plt.show()

    def inference_test_gpt_decoder(self, input_tokens, max_length=50):
        input_tokens = input_tokens.to(self.device)
        self.eval()
        generated_sequences = self.generate(input_tokens, max_length=max_length)
        print("Example Untrained GPTDecoder Prediction:", generated_sequences)

if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Instantiate the tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    # Set up the test input
    input_text = "Hello world"
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    mask = torch.ones(input_ids.shape)

    # Set up the GPTDecoder model
    model = GPTDecoder(
        n_layer=2,
        n_head=2,
        d_model=128,
        d_ff=256,
        vocab_size=tokenizer.vocab_size,
        max_seq_len=32,
        padding_token_id=tokenizer.pad_token_id,
        device = device,
        dropout=0.1
    )

    # Perform a forward pass on the input
    output = model(input_ids)

    # Verify that the output shape is correct
    assert output.shape == (1, input_ids.shape[1], tokenizer.vocab_size)
    print("Output shape is correct")

    # Verify that the output values are within a reasonable range
    assert torch.all(output >= -10) and torch.all(output <= 10), "Output values out of range"
    print("Output values are within a reasonable range")


    input_tokens = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]).cuda()
    padding_token_id = 0
    model = GPTDecoder(
        n_layer=2,
        n_head=2,
        d_model=128,
        d_ff=256,
        vocab_size=tokenizer.vocab_size,
        max_seq_len=32,
        padding_token_id=padding_token_id,
        device = device,
        dropout=0.1
    )

    combined_mask = model.create_combined_mask(input_tokens)
    causal_mask = combined_mask[0, 0].detach()

    model.plot_causal_mask(causal_mask)
    model.visualize_positional_encoding()

    input_tokens = torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    model.inference_test_gpt_decoder(input_tokens)



