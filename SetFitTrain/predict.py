from datasets import Dataset
from sklearn.metrics import classification_report
from numpy import mean
from datetime import datetime
from setfit import SetFitModel, Trainer, TrainingArguments
import re
import os
import json


def process_dataset(example):

    # concatenate title and body
    text = (example['title'] or "") + " " + (example['body'] or "")
    
    example['text'] = text
    return example

BASE_MODEL = "Collab-uniba/github-issues-mpnet-st-e10"
RANDOM_SEED = 42
OUTPUT_PATH = 'output'

ds = Dataset.from_csv({ "train": "data/nlbse_23_train.csv", "test": "data/nlbse_23_test.csv" })
ds["train"].to_pandas()

ds = ds.shuffle(seed=RANDOM_SEED)
ds = ds.map(process_dataset)
ds = ds.select_columns(['label', 'text'])

references = {}
predictions = {}

train_set = ds['train']

model = SetFitModel.from_pretrained(BASE_MODEL)

args = TrainingArguments(
    output_dir=f'{OUTPUT_PATH}',
    save_strategy="no",
    seed=RANDOM_SEED,
    batch_size=(16, 2),
    num_epochs=1,
    num_iterations=20,
)

# Track the training time

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_set,
)
start_time = datetime.now()
trainer.train()
end_time = datetime.now()
trainin_time = end_time - start_time
print(f"Training time: {trainin_time}")

test_set = ds['test']

references = list(test_set['label'])

# track the prediction time
start_time = datetime.now()
predictions = list(model.predict(test_set['text'], batch_size=8, show_progress_bar=True))
end_time = datetime.now()

prediction_time = end_time - start_time
print(f"Prediction time: {prediction_time}")

results = {}

results = classification_report(references, predictions, digits=4, output_dict=True)

results['training_time'] = trainin_time.total_seconds()
results['prediction_time'] = prediction_time.total_seconds()

output_file_name = 'results.json'


with open(os.path.join(OUTPUT_PATH, output_file_name), 'w') as fp:
    json.dump(results, fp, indent=2)

print(classification_report(references, predictions, digits=2))

