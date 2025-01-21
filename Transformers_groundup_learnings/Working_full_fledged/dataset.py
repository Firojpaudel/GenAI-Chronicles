import torch
import torch.nn as nn
from torch.utils.data import Dataset

class BilingualDataset(Dataset):
    def __init__(self, dataset, tokenizer_src, tokenizer_tgt, lang_src, lang_tgt, seq_len):
        self.dataset = dataset
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.lang_src = lang_src
        self.lang_tgt = lang_tgt
        self.seq_len = seq_len

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        src_text = item['translation'][self.lang_src]
        tgt_text = item['translation'][self.lang_tgt]

        # Transform the text into tokens
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        # Truncate tokens if they are too long
        if len(enc_input_tokens) > self.seq_len - 2:
            enc_input_tokens = enc_input_tokens[:self.seq_len - 2]
        if len(dec_input_tokens) > self.seq_len - 1:
            dec_input_tokens = dec_input_tokens[:self.seq_len - 1]

        # Add sos, eos and padding to each sentence
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2  # We will add <s> and </s>
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1  # We will only add <s>, and </s> only on the label

        enc_input_tokens = [self.tokenizer_src.token_to_id('[SOS]')] + enc_input_tokens + [self.tokenizer_src.token_to_id('[EOS]')] + [self.tokenizer_src.token_to_id('[PAD]')] * enc_num_padding_tokens
        dec_input_tokens = [self.tokenizer_tgt.token_to_id('[SOS]')] + dec_input_tokens + [self.tokenizer_tgt.token_to_id('[PAD]')] * dec_num_padding_tokens

        # Create the input and output tensors
        encoder_input = torch.tensor(enc_input_tokens, dtype=torch.long)
        decoder_input = torch.tensor(dec_input_tokens, dtype=torch.long)
        label = torch.tensor(dec_input_tokens[1:] + [self.tokenizer_tgt.token_to_id('[EOS]')], dtype=torch.long)

        return {
            'encoder_input': encoder_input,
            'decoder_input': decoder_input,
            'label': label,
            'encoder_mask': (encoder_input != self.tokenizer_src.token_to_id('[PAD]')).unsqueeze(0).unsqueeze(0),
            'decoder_mask': (decoder_input != self.tokenizer_tgt.token_to_id('[PAD]')).unsqueeze(0).unsqueeze(0)
        }
        
def causal_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0