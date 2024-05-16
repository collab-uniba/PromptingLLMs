from openai import OpenAI
from tqdm import tqdm
import json
import os
from loguru import logger
import time

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

def save_responses(responses, responses_path):
    with open(responses_path, 'w') as file:
        json.dump(responses, file, indent=4)

def process_prompts(responses_path, prompts, logger):

    try:
        with open(responses_path, 'r') as file:
            # Load the JSON data into a Python dictionary
            results = json.load(file)
    except:
        results = {}

    # Have each GPU do inference, prompt by prompt
    # prompts is a dict with keys as prompt ids and values as prompts
    for prompt_id, prompt in prompts.items():
        logger.info(f"Processing prompt {prompt_id}")
        response = client.chat.completions.create(
        model=MODEL_NAME,
        response_format={ "type": "json_object" },
        messages=[
            {
            "role": "user",
            "content": prompt['prompt']
            }
        ],
        temperature=TEMPERATURE,
        max_tokens=500,
        )
        # parse the response
        text_response = response.choices[0].message.content
        json_response = json.loads(text_response)
        
        # save the response
        results[prompt_id] = json_response

        # save the responses
        with open(responses_path, 'w') as file:
            # Write the JSON data to the file
            json.dump(results, file)

TEMPERATURE = 0.0
MODEL_NAME = 'gpt-4-turbo'

os.environ["OPENAI_API_KEY"] = "sk-proj-ACoBL2qoXFxDPq0hRoeFT3BlbkFJu43Cm8sDtKM9TakagFRA"

client = OpenAI(
  organization='org-vEijKmj1E9SRyolQlQov9fko',
  project='proj_bFDwVo6cD5P3jIUaZMYkx18f',
)

prompts_path = 'data/prompts.json'
responses_dir = os.path.join(*['responses', 'OpenAI', MODEL_NAME])
os.makedirs(responses_dir, exist_ok=True)

responses_path = os.path.join(responses_dir, "responses.json")

responses = load_responses(responses_path)

prompts = load_prompts(prompts_path, responses)

logger.info(f"Loaded {len(prompts)} prompts")

logger.info("Starting inference")

process_prompts(responses_path, prompts, logger)

logger.info("Inference complete")
