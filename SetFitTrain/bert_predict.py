from datasets import Dataset
import re
from datetime import datetime
import evaluate
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding
from transformers import RobertaTokenizerFast, RobertaConfig, RobertaForSequenceClassification, TextClassificationPipeline
import numpy as np
from sklearn.metrics import classification_report
from numpy import mean
import json
import os

def process_dataset(example):

    example['label'] = label2id[example['label']]

    # concatenate title and body
    text = (example['title'] or "") + " " + (example['body'] or "")

    # Remove strings between triple quotes
    text = re.sub(r'```.*?```', ' ', text, flags=re.DOTALL)

    # Remove new lines
    text = re.sub(r'\n', ' ', text)

    # Remove links
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', text)

    # Remove digits
    text = re.sub(r'\d+', ' ', text)

    # Remove special characters except the question marks
    text = re.sub(r'[^a-zA-Z0-9?\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)

    example['text'] = text

    return example


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels, average="weighted")

label2id = { "bug": 0, "feature": 1, "question": 2 }
id2label = { 0: "bug", 1: "feature", 2: "question" }

BASE_MODEL = "roberta-base"
RANDOM_SEED = 42
OUTPUT_PATH = 'output'

ds = Dataset.from_csv({ "train": "data/nlbse_24_train.csv", "test": "data/nlbse_24_test.csv" })

ds = ds.shuffle(seed=RANDOM_SEED)
ds = ds.map(process_dataset)
ds = ds.select_columns(['label', 'text'])


references = {}
predictions = {}

device="cuda"
truncation=True
padding="max_length"
max_length=512

tokenizer = RobertaTokenizerFast.from_pretrained(BASE_MODEL)

config = RobertaConfig.from_pretrained(BASE_MODEL, num_labels=3)

model = RobertaForSequenceClassification.from_pretrained(BASE_MODEL, config=config)

classifier = TextClassificationPipeline(model=model, tokenizer=tokenizer, device=device, truncation=truncation, padding=padding, max_length=max_length)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

def preprocess_function(examples):
    return tokenizer(examples['text'], truncation=truncation, padding=padding, max_length=max_length)

train_set = ds["train"]
train_set = train_set.map(preprocess_function, batched=True)
train_set.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

test_set = ds["test"]
references = [id2label[id] for id in test_set['label']]
print(references)
test_set = test_set.map(preprocess_function, batched=True)
test_set.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

training_args = TrainingArguments(
    output_dir=f"{OUTPUT_PATH}",
    seed=RANDOM_SEED,
    num_train_epochs=10,
    per_device_train_batch_size=16,
)

print(train_set)
print(test_set)

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=train_set,
    data_collator=data_collator,
)

# print training time
start = datetime.now()
trainer.train()
end = datetime.now()
print("Training time: ", end - start)

start = datetime.now()
predictions = classifier(test_set['text'])
end = datetime.now()
print("Prediction time: ", end - start)
predictions = [pred['label'] for pred in predictions]

print(predictions)

out2label = {"LABEL_0": "bug", "LABEL_1": "feature", "LABEL_2":"question"}
predictions = [out2label[pred] for pred in predictions]

results = {}

results = classification_report(references, predictions, digits=4, output_dict=True)

output_file_name = 'results.json'

with open(os.path.join(OUTPUT_PATH, output_file_name), 'w') as fp:
    json.dump(results, fp, indent=2)

print(classification_report(references, predictions, digits=4))
