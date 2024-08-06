from datasets import Dataset
from sklearn.metrics import classification_report
from numpy import mean
from datetime import datetime
from setfit import SetFitModel, Trainer, TrainingArguments, sample_dataset
import re
import os
import json
import random
from tqdm import tqdm
import matplotlib.pyplot as plt


def process_dataset(example):
    text = (example['title'] or "") + " " + (example['body'] or "")

    example['text'] = text
    return example

BASE_MODEL = "Collab-uniba/github-issues-mpnet-st-e10"
OUTPUT_PATH = 'output'
DATASET_NAME = 'nlbse_24'

ds = Dataset.from_csv({ "train": f"data/{DATASET_NAME}_train.csv", "test": f"data/{DATASET_NAME}_test.csv" })
ds["train"].to_pandas()

ds = ds.map(process_dataset)
ds = ds.select_columns(['label', 'text'])

tqdm_step_bar = tqdm(range(110, 201, 10), desc="Training 10 models for each step")

random_seeds = [1473, 7022, 3142, 7759, 491, 7496, 2749, 4971, 1158, 5476]
for step in tqdm_step_bar:
    tqdm_step_bar.set_description(f"Using {step} examples per class")
    tqdm_iter_bar = tqdm(range(10), desc="Training 10 models")

    # create output directory
    output_dir = f'{OUTPUT_PATH}/{DATASET_NAME}/step_{step}/'
    # if folder already exists, skip
    if os.path.exists(output_dir):
        continue
    os.makedirs(output_dir)
    for i in tqdm_iter_bar:
        tqdm_iter_bar.set_description(f"Iteration {i}")
        random_seed = random_seeds[i]
        sample_size = step
        sample_ds = sample_dataset(dataset=ds["train"], num_samples=sample_size, seed=random_seed, label_column='label')
        model = SetFitModel.from_pretrained(BASE_MODEL)



        args = TrainingArguments(
            output_dir=output_dir,
            save_strategy="no",
            seed=random_seed,
            batch_size=(16, 2),
            num_epochs=1,
            num_iterations=20,
        )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=sample_ds,
        )

        start_time = datetime.now()
        trainer.train()
        end_time = datetime.now()
        trainin_time = end_time - start_time
        print(f"Training time: {trainin_time}")

        test_set = ds['test']

        references = list(test_set['label'])

        start_time = datetime.now()
        predictions = list(model.predict(test_set['text'], batch_size=8, show_progress_bar=True))
        end_time = datetime.now()

        prediction_time = end_time - start_time
        print(f"Prediction time: {prediction_time}")

        results = {}

        results = classification_report(references, predictions, digits=4, output_dict=True)

        results['training_time'] = trainin_time.total_seconds()
        results['prediction_time'] = prediction_time.total_seconds()
        results['random_seed'] = random_seed

        results_file_name = f'results_{i}.json'

        with open(f'{output_dir}/{results_file_name}', 'w') as f:
            json.dump(results, f)

        # Save predictions
        predictions_file_name = f'predictions_{i}.json'

        with open(f'{output_dir}/{predictions_file_name}', 'w') as f:
            json.dump(predictions, f)

        print(classification_report(references, predictions, digits=2))

# Calculate average f1 micro and macro

step_results = []
for i in range(5, 200, 5):
    iter_results = []
    for j in range(10):
        with open(f'{OUTPUT_PATH}/{DATASET_NAME}/step_{i}/results_{j}.json', 'r') as f:
            iter_results.append(json.load(f))
    # Calculate average
    f1_macro = mean([r['macro avg']['f1-score'] for r in iter_results])
    try:
        f1_micro = mean([r['micro avg']['f1-score'] for r in iter_results])
    except KeyError:
        f1_micro = mean([r['accuracy'] for r in iter_results])
    step_results.append({
        'f1_macro': f1_macro,
        'f1_micro': f1_micro
    })

# Save results
with open(f'{OUTPUT_PATH}/{DATASET_NAME}/step_results.json', 'w') as f:
    json.dump(step_results, f)

