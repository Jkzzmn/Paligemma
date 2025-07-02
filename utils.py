from modeling_gemma import PaliGemmaForConditionalGeneration, PaliGemmaConfig
from transformers import AutoTokenizer
import json
import glob
from safetensors import safe_open
from typing import Tuple
import os

def load_hf_model(model_path: str, device: str) -> Tuple[PaliGemmaForConditionalGeneration, AutoTokenizer]:
    print(f"Loading model from {model_path}")
    print(f"Target device: {device}")
    
    # Load the tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="right")
        print(f"Tokenizer loaded successfully. Vocabulary size: {len(tokenizer)}")
        assert tokenizer.padding_side == "right"
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        raise

    # Find all the *.safetensors files
    try:
        safetensors_files = glob.glob(os.path.join(model_path, "*.safetensors"))
        print(f"Found {len(safetensors_files)} safetensors files: {safetensors_files}")
    except Exception as e:
        print(f"Error finding safetensors files: {e}")
        raise

    # ... and load them one by one in the tensors dictionary
    tensors = {}
    try:
        for safetensors_file in safetensors_files:
            print(f"Loading tensors from {safetensors_file}")
            with safe_open(safetensors_file, framework="pt", device="cpu") as f:
                keys = f.keys()
                print(f"Found {len(keys)} keys in {safetensors_file}")
                for key in keys:
                    tensors[key] = f.get_tensor(key)
        print(f"Loaded {len(tensors)} tensors in total")
    except Exception as e:
        print(f"Error loading tensors: {e}")
        raise

    # Load the model's config
    try:
        config_path = os.path.join(model_path, "config.json")
        print(f"Loading config from {config_path}")
        with open(config_path, "r") as f:
            model_config_file = json.load(f)
            config = PaliGemmaConfig(**model_config_file)
        print(f"Config loaded successfully: {config}")
    except Exception as e:
        print(f"Error loading config: {e}")
        raise

    # Create the model using the configuration
    try:
        print("Creating model from config")
        model = PaliGemmaForConditionalGeneration(config)
        model.load_state_dict(tensors, strict=False)
        model = model.to(device)
        print(f"Model created successfully. Num parameters: {sum(p.numel() for p in model.parameters())}")
    except Exception as e:
        print(f"Error creating model: {e}")
        raise

    # Load the state dict of the model
    try:
        print("Loading state dict into model")
        model.load_state_dict(tensors, strict=False)
        print("State dict loaded successfully")
    except Exception as e:
        print(f"Error loading state dict: {e}")
        raise

    # Tie weights
    try:
        print("Tying weights")
        model.tie_weights()
        print("Weights tied successfully")
    except Exception as e:
        print(f"Error tying weights: {e}")
        raise

    print("Model loading complete")
    return (model, tokenizer)