import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Tuple

config = {
    "model_name": "mistralai/Mistral-7B-v0.3",
    "device_map": "auto",
    "torch_dtype": torch.bfloat16
}

def load_base_model(
    model_name: str = config["model_name"],
    device_map: str = config["device_map"],
    torch_dtype: str = config["torch_dtype"]
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device_map,
        torch_dtype=torch_dtype
    )
    model.gradient_checkpointing_enable()
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model loaded: {total_params/1e9:.2f}B total params")
    print(f"Trainable: {trainable_params/1e9:.2f}B params")

    return model, tokenizer

if __name__=="__main__":
    model, tokenizer = load_base_model()
