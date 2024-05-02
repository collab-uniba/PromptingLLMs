
from loguru import logger
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from accelerate import Accelerator
from accelerate.utils import gather_object
import pandas as pd
from tqdm import tqdm
import yaml
import re
import json
import os
import time

def load_prompts(prompts_path, responses=None):
    with open(prompts_path, 'r') as file:
        prompts = json.load(file)
    if responses:
        prompts = {k: v for k, v in prompts.items() if k not in responses}

def load_responses(responses_path):
    with open(responses_path, 'r') as file:
        responses = json.load(file)
    return responses

def process_prompts(accelerator, tokenizer, model, prompts_all, logger):
    # Sync GPUs and start the timer
    accelerator.wait_for_everyone()
    start = time.time()

    # Divide the prompt list onto the available GPUs 
    with accelerator.split_between_processes(prompts_all) as prompts:
        # Store output of generations in dict
        results = dict(outputs=[], num_tokens=0)

        # Have each GPU do inference, prompt by prompt
        for prompt in prompts:
            prompt_tokenized = tokenizer(prompt, return_tensors="pt").to("cuda")
            output_tokenized = model.generate(**prompt_tokenized, max_new_tokens=500)[0]

            # Remove prompt from output 
            output_tokenized = output_tokenized[len(prompt_tokenized["input_ids"][0]):]

            # Store outputs and number of tokens in results{}
            results["outputs"].append(tokenizer.decode(output_tokenized))
            results["num_tokens"] += len(output_tokenized)

        results = [results]  # Transform to list, otherwise gather_object() will not collect correctly

    # Collect results from all the GPUs
    results_gathered = gather_object(results)

    if accelerator.is_main_process:
        timediff = time.time() - start
        num_tokens = sum([r["num_tokens"] for r in results_gathered])

        logger.info(f"tokens/sec: {num_tokens // timediff}, time {timediff}, total tokens {num_tokens}, total prompts {len(prompts_all)}")
    return results_gathered

def save_responses(responses, responses_path):
    with open(responses_path, 'w') as file:
        json.dump(responses, file, indent=4)
