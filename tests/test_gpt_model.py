import torch
from gpt_model.gpt_decoder_layer import GPTDecoderLayer
from transformers import GPT2Tokenizer
from gpt_model.gpt_decoder import GPTDecoder

def test_gpt_decoder_layer():
    d_model = 256
    n_head = 8
    d_ff = 1024
    device = torch.device("cpu")

    decoder_layer = GPTDecoderLayer(d_model=d_model, n_head=n_head, d_ff=d_ff, device=device)

    x = torch.randn(4, 16, d_model)
    mask = torch.ones(4, 1, 16, dtype=torch.bool)

    out = decoder_layer(x, mask)

    assert out.shape == (4, 16, d_model), "Output shape doesn't match expected shape"

    print("GPTDecoderLayer tests passed.")

def test_gpt_decoder():
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

if __name__ == '__main__':
    test_gpt_decoder_layer()
    test_gpt_decoder()