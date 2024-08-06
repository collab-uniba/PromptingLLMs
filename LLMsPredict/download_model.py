# download_model.py
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import sys

def setup_transformers(model_name):
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Load model with bfloat16 precision
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
        )

        print("Tokenizer and model setup completed successfully.")
    except Exception as e:
        print(f"Error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python setup_transformers.py <model_name>")
        sys.exit(1)
    
    model_name = sys.argv[1]
    setup_transformers(model_name)

