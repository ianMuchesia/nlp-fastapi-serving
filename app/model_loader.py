import json
import torch
from app.model import SentimentModel

def get_model(config_path="checkpoints/config.json"):
    # 1. Load the Configuration
    with open(config_path, "r") as f:
        config = json.load(f)
        
    # 2. Load Vocab (still needed for the actual words)
    with open("checkpoints/vocab.json", "r") as f:
        vocab = json.load(f)
        
    # 3. Instantiate using values from the config 🛠️
    model = SentimentModel(
        vocab_size=config["vocab_size"],
        embedding_dim=config["embedding_dim"]
    )
    
    # 4. Load weights using the path from the config
    model.load_state_dict(
        torch.load(config["model_path"], map_location="cpu")
    )
    
    model.eval()
    
    
    return model, vocab