
from loguru import logger
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from accelerate import Accelerator
from accelerate.utils import gather_object
import yaml
import json
import os
import time
from math import ceil

def load_prompts(prompts_path, responses=None):
    with open(prompts_path, 'r') as file:
        prompts = json.load(file)
    if responses:
        prompts = {k: v for k, v in prompts.items() if k not in responses}
    return prompts

def load_responses(responses_path):
    if os.path.exists(responses_path):
        with open(responses_path, 'r') as file:
            responses = json.load(file)
    else:
        responses = {}
    return responses

def process_prompts(accelerator, tokenizer, model, prompts_all, logger, responses, responses_path, save_every=32):
    # Sync GPUs and start the timer
    accelerator.wait_for_everyone()
    start = time.time()

    num_batches = ceil(len(prompts_all) / save_every)
    results = dict(outputs={}, num_tokens=0)
    keys = list(prompts_all.keys())
    for batch_idx in range(num_batches):
        start_idx = batch_idx * save_every
        end_idx = start_idx + save_every
        batch_ids = keys[start_idx:end_idx]
        batch_prompts = {k: prompts_all[k] for k in batch_ids}

        # Divide the prompt list onto the available GPUs 
        with accelerator.split_between_processes(batch_prompts) as prompts:
            # Have each GPU do inference, prompt by prompt
            # prompts is a dict with keys as prompt ids and values as prompts
            for prompt_id, prompt in prompts.items():
                prompt_tokenized = tokenizer(prompt['prompt'], return_tensors="pt").to("cuda")
                output_tokenized = model.generate(**prompt_tokenized, max_new_tokens=500)[0]

                # Remove prompt from output 
                output_tokenized = output_tokenized[len(prompt_tokenized["input_ids"][0]):]

                # Store outputs and number of tokens in results{}
                results["outputs"][prompt_id] = (tokenizer.decode(output_tokenized))
                results["num_tokens"] += len(output_tokenized)

            results = [results]  # Transform to list, otherwise gather_object() will not collect correctly

        # Collect results from all the GPUs
        results_gathered = gather_object(results)
        for prompt_id, response in results_gathered[0]['outputs'].items():
            responses[prompt_id] = response
        save_responses(responses, responses_path)

        # Free up results and save number of tokens
        results = dict(outputs={}, num_tokens=results_gathered[0]["num_tokens"])

    if accelerator.is_main_process:
        timediff = time.time() - start
        num_tokens = sum([r["num_tokens"] for r in results_gathered])

        logger.info(f"tokens/sec: {num_tokens // timediff}, time {timediff}, total tokens {num_tokens}, total prompts {len(prompts_all)}")
    return results_gathered[0]

def save_responses(responses, responses_path):
    with open(responses_path, 'w') as file:
        json.dump(responses, file, indent=4)

logger.add("predict.log")

config_path = "config.yaml"
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

accelerator = Accelerator()
nf4_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_quant_type="nf4",
   bnb_4bit_use_double_quant=True,
   bnb_4bit_compute_dtype=torch.bfloat16
)

logger.info(f"Using model: {config['model_name']}")
logger.info(f"Loding model and tokenizer")

tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
model = AutoModelForCausalLM.from_pretrained(
    config["model_name"],    
    device_map={"": accelerator.process_index},
    torch_dtype=torch.bfloat16,
    quantization_config=nf4_config
)

logger.info(f"Model and tokenizer loaded")

responses_dir = os.path.join(*[config["responses_dir"], config["model_name"]])
os.makedirs(responses_dir, exist_ok=True)

responses_path = os.path.join(responses_dir, "responses.json")

responses = load_responses(responses_path)
prompts = load_prompts(config["prompts_path"], responses)

logger.info(f"Loaded {len(prompts)} prompts")

logger.info("Starting inference")

results = process_prompts(accelerator, tokenizer, model, prompts, logger, responses, responses_path)

logger.info("Inference complete")

logger.info(f"Saving responses to {responses_path}")
save_responses(responses, responses_path)


