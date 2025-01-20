import torch 
import torch.nn as nn
from pathlib import Path
from datasets import load_dataset
from transformers import Tokenizer
from tokenizers.models import WordLevel 
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace


def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]
        
def create_tokenizer(config, ds, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer= WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency =2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer= trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_dataset(config):
    ds_train = load_dataset("cfilt/iitb-english-hindi", 'en-hi', split= 'train')
    ds_valid = load_dataset("cfilt/iitb-english-hindi", 'en-hi', split= 'validation')
    ds_test = load_dataset("cfilt/iitb-english-hindi", 'en-hi', split= 'test')
    
    ## Tokenizers 
    tokenizer_en = create_tokenizer(config, ds_train, 'en')
    tokenizer_hi = create_tokenizer(config, ds_train, 'hi')
    
    
    