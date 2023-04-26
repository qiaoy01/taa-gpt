import torch
from gpt_model.gpt_decoder_layer import GPTDecoderLayer
from transformers import AutoTokenizer
from gpt_model.gpt_decoder import GPTDecoder
import json
import seaborn as sns
from scipy.stats import entropy
import matplotlib.pyplot as plt
import numpy as np
from torchviz import make_dot
from graphviz import Digraph

def test_gpt_decoder_layer():
    d_model = 256
    n_head = 8
    d_ff = 1024
    device = torch.device("cpu")

    decoder_layer = GPTDecoderLayer(d_model=d_model, n_head=n_head, d_ff=d_ff, device=device)

    x = torch.randn(4, 16, d_model)
    mask = torch.ones(4, 1, 16, dtype=torch.bool)

    out, _ = decoder_layer(x, mask)

    assert out.shape == (4, 16, d_model), "Output shape doesn't match expected shape"

    print("GPTDecoderLayer tests passed.")

def test_gpt_decoder():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load training parameters from config file
    with open('config.json', 'r') as f:
        config = json.load(f)

    # Instantiate the tokenizer
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.add_special_tokens(config['special_tokens'])
    
    # Set up the test input
    input_text = "Hello world"
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    mask = torch.ones(input_ids.shape)

    # Set up the GPTDecoder model
    model = GPTDecoder(
        vocab_size=tokenizer.vocab_size,
        d_model=config['d_model'],
        n_head=config['n_head'],
        d_ff=config['d_ff'],
        n_layer=config['n_layer'],
        max_seq_len=config['seq_len'],
        dropout=config['dropout'],
        padding_token_id=tokenizer.pad_token_id,
        device=device
    )

    # Perform a forward pass on the input
    output = model(input_ids)

    # Verify that the output shape is correct
    assert output.shape == (1, input_ids.shape[1], tokenizer.vocab_size)
    print("Output shape is correct")

    # Verify that the output values are within a reasonable range
    assert torch.all(output >= -10) and torch.all(output <= 10), "Output values out of range"
    print("Output values are within a reasonable range")


    input_tokens = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]]).cuda()

    combined_mask = model.create_combined_mask(input_tokens)
    causal_mask = combined_mask[0, 0].detach()

    model.plot_causal_mask(causal_mask)
    model.visualize_positional_encoding()

    input_tokens = torch.LongTensor([[15496, 3228, 314, 1101, 10948, 73, 509, 5117, 10036, 13, 632, 338, 33160, 13, 266, 405, 83, 1058, 35, 12520, 97, 245]])
    generated_sequences = model.inference_test_gpt_decoder(input_tokens)

    generated_text = tokenizer.decode(generated_sequences[0][0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    print("Example Untrained GPTDecoder text:", generated_text)

def test_untrained_distribution():

    # Set the device to use for computations
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Sample input sequence (batch_size=1, seq_len=10)
    input_tokens = torch.tensor([[2, 6, 9, 4, 1, 8, 7, 3, 5, 0]]).to(device)

    # Hyperparameters for the GPTDecoder
    vocab_size = 11
    d_model = 64
    n_head = 4
    d_ff = 128
    n_layer = 6
    max_seq_len = 50
    padding_token_id = 0
    dropout = 0.1

    # Instantiate the GPTDecoder
    gpt_decoder = GPTDecoder(vocab_size, d_model, n_head, d_ff, n_layer, max_seq_len, padding_token_id, device, dropout).to(device)

    # Pass the input_tokens to the GPTDecoder
    output = gpt_decoder(input_tokens)

    # Check the output dimensions
    print(f"Output shape: {output.shape}")  # Should be (batch_size, seq_len, vocab_size)

    # Inspect the distribution of values in the output tensor
    print(f"Output mean: {output.mean().item()}, Output std: {output.std().item()}")

    # Visualize the output using a heatmap


    output_np = output.squeeze(0).detach().cpu().numpy()
    plt.figure(figsize=(10, 6))
    sns.heatmap(output_np, annot=False, cmap="coolwarm")
    plt.xlabel("Vocabulary")
    plt.ylabel("Sequence")
    plt.title("Output Distribution Heatmap")
    plt.show()


    # Sample input sequence (batch_size=1, seq_len=50)
    input_tokens = torch.randint(0, 100, (1, 50)).to(device)

    # Hyperparameters for the GPTDecoder
    vocab_size = 100
    d_model = 256
    n_head = 8
    d_ff = 512
    n_layer = 12
    max_seq_len = 200
    padding_token_id = 0
    dropout = 0.1

    # Instantiate the GPTDecoder
    gpt_decoder = GPTDecoder(vocab_size, d_model, n_head, d_ff, n_layer, max_seq_len, padding_token_id, device, dropout).to(device)

    # Pass the input_tokens to the GPTDecoder
    output = gpt_decoder(input_tokens)

    # Compute the output probabilities using softmax
    output_prob = torch.softmax(output, dim=-1)

    # Check the output dimensions
    print(f"Output shape: {output_prob.shape}")  # Should be (batch_size, seq_len, vocab_size)

    # Compute the entropy for each position in the sequence
    entropies = np.array([entropy(output_prob[0, i, :].cpu().detach().numpy()) for i in range(output_prob.shape[1])])

    # Calculate the average entropy
    average_entropy = np.mean(entropies)
    print(f"Average entropy: {average_entropy}")

    # Define a threshold for the entropy (e.g., 90% of the maximum possible entropy)
    entropy_threshold = 0.9 * np.log(vocab_size)
    print(f"Entropy threshold: {entropy_threshold}")

    # Assess whether the output distribution is reasonable
    distribution_reasonable = average_entropy >= entropy_threshold
    print(f"Is the output distribution reasonable? {distribution_reasonable}")

    # Load training parameters from config file
    with open('config.json', 'r') as f:
        config = json.load(f)

    # Use GPT2Tokenizer to tokenize the input text
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.add_special_tokens(config['special_tokens'])

    text = "The quick brown fox jumps over the lazy dog."
    input_tokens = tokenizer.encode(text, return_tensors="pt").to(device)

    # Remove the last token to predict it using the GPTDecoder
    input_tokens = input_tokens[:, :-1]

    # Hyperparameters for the GPTDecoder
    vocab_size = tokenizer.vocab_size
    d_model = 768
    n_head = 12
    d_ff = 3072
    n_layer = 12
    max_seq_len = 512
    padding_token_id = tokenizer.pad_token_id
    dropout = 0.1

    # Instantiate the GPTDecoder
    gpt_decoder = GPTDecoder(vocab_size, d_model, n_head, d_ff, n_layer, max_seq_len, padding_token_id, device, dropout).to(device)

    # Pass the input_tokens to the GPTDecoder
    output_logits = gpt_decoder(input_tokens)

    # Compute the probabilities using softmax
    output_prob = torch.softmax(output_logits, dim=-1)

    # Get the logits and probabilities for the last token in the input sequence
    last_token_logits = output_logits[0, -1, :].cpu().detach().numpy()
    last_token_prob = output_prob[0, -1, :].cpu().detach().numpy()

    # Get the top 5 predicted tokens and their probabilities
    top_k = 5
    top_predicted_tokens = last_token_prob.argsort()[-top_k:][::-1]
    top_predicted_probs = last_token_prob[top_predicted_tokens]

    # Ground truth token (the actual next token in the text)
    ground_truth_token = tokenizer.encode(text)[-1]

    print(f"Input text: {text}")
    print(f"Last token in the input sequence: {tokenizer.decode(input_tokens[0, -1].item())}")
    print(f"Ground truth token: {tokenizer.decode(ground_truth_token)}")
    print("Top predicted tokens and their probabilities:")
    for token, prob in zip(top_predicted_tokens, top_predicted_probs):
        print(f"{tokenizer.decode(token)}: {prob:.4f}")

    # Check if the ground truth token is among the top predicted tokens
    correct_prediction = ground_truth_token in top_predicted_tokens
    print(f"Is the ground truth token among the top {top_k} predicted tokens? {correct_prediction}")

def virtualize_attention():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open('config.json', 'r') as f:
        config = json.load(f)

    # Use GPT2Tokenizer to tokenize the input text
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.add_special_tokens(config['special_tokens'])


    # Hyperparameters for the GPTDecoder
    vocab_size = tokenizer.vocab_size
    d_model = 64
    n_head = 8
    d_ff = 128
    n_layer = 3
    max_seq_len = 128
    padding_token_id = tokenizer.pad_token_id
    dropout = 0.1

    # Instantiate the GPTDecoder
    gpt_decoder = GPTDecoder(vocab_size, d_model, n_head, d_ff, n_layer, max_seq_len, padding_token_id, device, dropout).to(device)
    text_input = "The quick brown fox jumps over the lazy dog."
    tokenized_input = tokenizer.encode(text_input)

    input_tokens = torch.tensor(tokenized_input, dtype=torch.long).unsqueeze(0)
    gpt_decoder.visualize_attention(input_tokens)

    print_model_structure(gpt_decoder)

def print_model_structure(model):
    print("Model Structure:")
    for name, module in model.named_modules():
        print(f"{name}: {module.__class__.__name__}")
    
    input_tensor = torch.randint(0, 100, (1, 10)).long()

    # Perform a forward pass through the model with the input tensor
    output = model(input_tensor)

    # Visualize the model structure
    make_dot(output, params=dict(model.named_parameters())).render("model_structure", format="png")

    # Print the model structure
    def print_model_structure(module, indent=0):
        for name, sub_module in module.named_children():
            print("  " * indent + name + ": " + sub_module.__class__.__name__)
            print_model_structure(sub_module, indent + 1)

    print("Model Structure:")
    print_model_structure(model)



def draw_gpt_decoder_structure():
    dot = Digraph(comment='GPTDecoder Model Structure', format='png')

    # GPTDecoder
    dot.node('A', 'GPTDecoder')
    
    # Embedding
    dot.node('B', 'embedding: Embedding')
    dot.edge('A', 'B')

    # Layers
    dot.node('C', 'layers: ModuleList')
    dot.edge('A', 'C')

    # Layer 0
    draw_gpt_decoder_layer(dot, '0', 'D')
    
    # Layer 1
    draw_gpt_decoder_layer(dot, '1', 'E')
    
    # Layer 2
    draw_gpt_decoder_layer(dot, '2', 'F')

    # Final Linear Layer
    dot.node('G', 'fc: Linear')
    dot.edge('A', 'G')

    dot.render('gpt_decoder_structure', view=True)

def draw_gpt_decoder_layer(dot, layer_idx, node_prefix):
    layer = f'layers.{layer_idx}'
    layer_label = f'{layer}: GPTDecoderLayer'
    dot.node(f'{node_prefix}', layer_label)
    dot.edge('C', f'{node_prefix}')

    # Self-Attention
    draw_self_attention(dot, layer, node_prefix)

    # Feed-Forward
    draw_feed_forward(dot, layer, node_prefix)

    # Layer Norms and Dropout
    dot.node(f'{node_prefix}LN1', f'{layer}.layer_norm1: LayerNorm')
    dot.edge(f'{node_prefix}', f'{node_prefix}LN1')
    dot.node(f'{node_prefix}LN2', f'{layer}.layer_norm2: LayerNorm')
    dot.edge(f'{node_prefix}', f'{node_prefix}LN2')
    dot.node(f'{node_prefix}DO', f'{layer}.dropout: Dropout')
    dot.edge(f'{node_prefix}', f'{node_prefix}DO')

def draw_self_attention(dot, layer, node_prefix):
    sa = f'{layer}.self_attention'
    dot.node(f'{node_prefix}SA', f'{sa}: MultiHeadAttention')
    dot.edge(f'{node_prefix}', f'{node_prefix}SA')
    
    # Linear layers
    dot.node(f'{node_prefix}SAWQ', f'{sa}.w_qs: Linear')
    dot.edge(f'{node_prefix}SA', f'{node_prefix}SAWQ')
    dot.node(f'{node_prefix}SAWK', f'{sa}.w_ks: Linear')
    dot.edge(f'{node_prefix}SA', f'{node_prefix}SAWK')
    dot.node(f'{node_prefix}SAWV', f'{sa}.w_vs: Linear')
    dot.edge(f'{node_prefix}SA', f'{node_prefix}SAWV')

    # FC and Attention
    dot.node(f'{node_prefix}SAFC', f'{sa}.fc: Linear')
    dot.edge(f'{node_prefix}SA', f'{node_prefix}SAFC')
    dot.node(f'{node_prefix}SAA', f'{sa}.attention: ScaledDotProductAttention')
    dot.edge(f'{node_prefix}SA', f'{node_prefix}SAA')

    # Dropout and Softmax
    dot.node(f'{node_prefix}SADO', f'{sa}.attention.dropout: Dropout')
    dot.edge(f'{node_prefix}SAA', f'{node_prefix}SADO')
    dot.node(f'{node_prefix}SASM', f'{sa}.attention.softmax: Softmax')
    dot.edge(f'{node_prefix}SAA', f'{node_prefix}SASM')

    # Self-Attention Dropout and LayerNorm
    dot.node(f'{node_prefix}SAD', f'{sa}.dropout: Dropout')
    dot.edge(f'{node_prefix}SA', f'{node_prefix}SAD')
    dot.node(f'{node_prefix}SALN', f'{sa}.layer_norm: LayerNorm')
    dot.edge(f'{node_prefix}SA', f'{node_prefix}SALN')

def draw_feed_forward(dot, layer, node_prefix):
    ff = f'{layer}.feed_forward'
    dot.node(f'{node_prefix}FF', f'{ff}: PositionwiseFeedForward')
    dot.edge(f'{node_prefix}', f'{node_prefix}FF')

    # Feed-Forward Sequential
    dot.node(f'{node_prefix}FFS', f'{ff}.fc: Sequential')
    dot.edge(f'{node_prefix}FF', f'{node_prefix}FFS')

    # Linear, GELU, Dropout, Linear, LayerNorm
    dot.node(f'{node_prefix}FFSL1', f'{ff}.fc.0: Linear')
    dot.edge(f'{node_prefix}FFS', f'{node_prefix}FFSL1')
    dot.node(f'{node_prefix}FFSG', f'{ff}.fc.1: GELU')
    dot.edge(f'{node_prefix}FFS', f'{node_prefix}FFSG')
    dot.node(f'{node_prefix}FFSD', f'{ff}.fc.2: Dropout')
    dot.edge(f'{node_prefix}FFS', f'{node_prefix}FFSD')
    dot.node(f'{node_prefix}FFSL2', f'{ff}.fc.3: Linear')
    dot.edge(f'{node_prefix}FFS', f'{node_prefix}FFSL2')
    dot.node(f'{node_prefix}FFSLN', f'{ff}.fc.4: LayerNorm')
    dot.edge(f'{node_prefix}FFS', f'{node_prefix}FFSLN')

if __name__ == '__main__':
    test_gpt_decoder_layer()
    test_gpt_decoder()
    test_untrained_distribution()
    virtualize_attention()
    draw_gpt_decoder_structure()