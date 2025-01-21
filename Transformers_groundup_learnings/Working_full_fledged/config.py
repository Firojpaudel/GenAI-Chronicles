from pathlib import Path

def get_config():
    return {
        "batch_size": 32,
        "num_epochs": 20,
        "lr": 10**-4,
        "seq_len": 128,
        "d_model": 512,
        "datasource": 'cfilt/iitb-english-hindi',
        "lang_src": "en",
        "lang_tgt": "hi",
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": "latest",
        "tokenizer_file": "tokenizers_here_/tokenizer_{0}.json",
        "experiment_name": "runs/tmodel"
    }

def get_weights_file_path(config, epoch: str):
    model_folder = Path(config['datasource']) / config['model_folder']
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(model_folder / model_filename)

# Find the latest weights file in the weights folder
def latest_weights_file_path(config):
    model_folder = Path(config['datasource']) / config['model_folder']
    model_filename = f"{config['model_basename']}*"
    weights_files = list(model_folder.glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])
