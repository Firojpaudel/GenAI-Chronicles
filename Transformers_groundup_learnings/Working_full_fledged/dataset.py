import torch 
import torch.nn as nn
from torch.utils.data import Dataset

class BilingualDataset(Dataset):
    def __init__(self, ds, tokenizer_en, tokenizer_hi, src_lang='en', tgt_lang='hi', seq_len=128):
        self.seq_len = seq_len
        self.ds = ds
        self.tokenizer_en = tokenizer_en
        self.tokenizer_hi = tokenizer_hi

        self.sos_token = torch.Tensor([tokenizer_hi.token_to_id(["[SOS]"])], dtype=torch.int64)
        self.eos_token = torch.Tensor([tokenizer_hi.token_to_id(["[EOS]"])], dtype=torch.int64)
        self.pad_token = torch.Tensor([tokenizer_hi.token_to_id(["[PAD]"])], dtype=torch.int64)
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, idx: Any) -> Any:
        item = self.ds[idx]
        src = item['translation']['en']
        tgt = item['translation']['hi']
        
        enc_input_tokens = self.tokenizer_en.encode(src).ids
        dec_input_tokens = self.tokenizer_hi.encode(tgt).ids
        
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1
        
        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError("Sequence length is too long")
        
        ## Adding SOS and EOS to the source text
        encoder_input = torch.cat(
            [
                self.sos_token, 
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64)
            ]
        )
        ## Adding SOS to the decoder input
        decoder_input= torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token]* dec_num_padding_tokens, dtype=torch.int64)
            ]
        )
        ## Adding EOS to the label 
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype= torch.int64), 
                self.eos_token, 
                torch.tensor([self.pad_token]* dec_num_padding_tokens, dtype=torch.int64)
            ]
        )
        
        # Double checking the size of the tensors to make sure they are all seq_len long
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len
        
        return{
            'encoder_input': encoder_input,
            'decoder_input': decoder_input,   
            'encoder_mask': (encoder_input != self.pad_token).unsqueeze(0).int(),
            'decoder_mask': (decoder_input != self.pad_token).unsqueeze(0).int() & casual_mask(decoder_input.size(0)),
            'label': label, 
            'src_text': "en", 
            'tgt_text': "hi"
        }

def casual_mask(size):
    mask= torch.triu(torch.ones(size, size), diagonal=1).type(torch.int)
    return mask == 0   