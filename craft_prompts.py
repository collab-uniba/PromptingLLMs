# This scripts generates prompts to be used by LLMs for issue report classification
# The prompts are generated based on the dataset and the prompt template
# The prompts are saved in a file

import pandas as pd
import yaml
import json
from tqdm import tqdm

# Load the dataset
def load_dataset(dataset_path):
    dataset = pd.read_csv(dataset_path)
    return dataset

# Load the prompt template
def load_prompt_template(template_path):
    with open(template_path, 'r') as file:
        template = yaml.safe_load(file)
    return template

def craft_prompt(template, replacements):
    task_prompt = replace_placeholders(template['task'], replacements)
    return '\n'.join([task_prompt, template['label_explanations'], template['format_instructions'], template['output']])

# Replace placeholders in a template string with values from the dataset
def replace_placeholders(template, replacements):
    prompt = template.format(**replacements)
    return prompt

# Generate prompts for each row in the dataset
def generate_prompts(dataset, template, columns, target_column):
    prompts = {}
    # Each prompt should have index as key and dict of columns as value
    for index, row in tqdm(dataset.iterrows()):
        replacements = {}
        for column in columns:
            replacements[column] = row[column]
        prompt = craft_prompt(template, replacements)
        # Add the target column value to the prompt
        test_pair = {'prompt': prompt, 'target': row[target_column]}
        prompts[index] = test_pair
    return prompts

# Save the prompts to a json file
def save_prompts(prompts, output_path):
    with open(output_path, 'w') as file:
        json.dump(prompts, file, indent=4)


with open("config.yaml", 'r') as file:
    params = yaml.safe_load(file)
dataset = load_dataset(params['data_path'])
template = load_prompt_template(params['template_path'])
prompts = generate_prompts(dataset, template, params['column_names'], params['target_column'])
save_prompts(prompts, params['prompts_path'])

