import os
import json
import re
import yaml
from openpyxl import Workbook
from sklearn.metrics import classification_report
import sys

sys.path.append('externals/sklearn-cls-report2excel')
from convert_report2excel import convert_report2excel


def get_response_paths(responses_dir):
    response_paths = []
    for folder in os.listdir(responses_dir):
        folder_path = os.path.join(responses_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        for subfolder in os.listdir(folder_path):
            response_path = os.path.join(folder_path, subfolder, "responses.json")
            response_paths.append(response_path)
    return response_paths


def get_model_name(response_path):
    return response_path.split("/")[-2]


def get_label(text):
    try:
        label = re.search(r"(:?\\\"|\")label(:?\\\"|\"):\s*(:?\\\"|\")(bug|feature|documentation|question)(:?\\\"|\")", text, flags=re.DOTALL)[4]
        return label
    except Exception:
        return ""


def get_predictions(response_path):
    with open(response_path, 'r') as file:
        responses = json.load(file)
    predictions = {}
    for prompt_id, response in responses.items():
        predictions[prompt_id] = get_label(response)
    return predictions


def get_true_labels(prompts_path):
    with open(prompts_path, 'r') as file:
        prompts = json.load(file)
    true_labels = {}
    for prompt_id, prompt_data in prompts.items():
        true_labels[prompt_id] = prompt_data["target"]
    return true_labels


def evaluate_model(responses_dir, prompts_path):
    response_paths = get_response_paths(responses_dir)
    true_labels = get_true_labels(prompts_path)
    metrics = {}
    for response_path in response_paths:
        model_name = get_model_name(response_path)
        predictions = get_predictions(response_path)
        true_labels = get_true_labels(prompts_path)
        y_true = []
        y_pred = []
        for prompt_id, true_label in true_labels.items():
            y_true.append(true_label)
            y_pred.append(predictions[prompt_id])
        report = classification_report(y_true, y_pred, labels=['bug', 'documentation', 'feature', 'question'], output_dict=True)
        metrics[model_name] = report
    return metrics


def create_excel_table(metrics, output_path):
    # Create a workbook with a sheet for each model
    wb = Workbook()
    wb.remove(wb.active)
    for model_name, metric in metrics.items():
        convert_report2excel(
            workbook=wb,
            report=metric,
            sheet_name=model_name,
        )
    wb.save(os.path.join(output_path, "report.xlsx"))


responses_dir = "responses"
config_path = "config.yaml"
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)
prompts_path = config["prompts_path"]
metrics = evaluate_model(responses_dir, prompts_path)
create_excel_table(metrics, responses_dir)


        

