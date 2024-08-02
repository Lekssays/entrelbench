import json
from collections import Counter, defaultdict
from typing import Dict, List
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import glob
import ast
import argparse


def safe_eval(s):
    try:
        return ast.literal_eval(s)
    except:
        # If parsing fails, return the original string
        return s


def calculate_metrics_EntExt(data: List[Dict[str, str]]) -> Dict[str, float]:
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for item in data:
        gold = set([e.lower() for e in safe_eval(item['gold'])])
        pred = set([e.lower() for e in safe_eval(item['predicted'])])
        
        true_positives += len(gold.intersection(pred))
        false_positives += len(pred - gold)
        false_negatives += len(gold - pred)

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': true_positives,
        'fp': false_positives,
        'fn': false_negatives
    }


def calculate_metrics_RelExt(data: List[Dict[str, List[str]]]) -> Dict[str, float]:
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for item in data:
        gold_relations = set([e.lower() for e in item['gold']])
        predicted_relations = set([e.lower() for e in item['predicted']])
        
        true_positives += len(gold_relations.intersection(predicted_relations))
        false_positives += len(predicted_relations - gold_relations)
        false_negatives += len(gold_relations - predicted_relations)

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': true_positives,
        'fp': false_positives,
        'fn': false_negatives,
    }


def write_markdown(results: Dict[str, Dict[str, Dict]], filename: str, task: str, dataset: str):
    with open(filename, 'w') as f:
        f.write(f"# Results for Task {task}\n\n")
        for model, metrics in results.items():
            f.write(f"## {model} on Dataset {dataset}\n\n")
            if task == 'EntExt':
                f.write(f"\n### Overall Metrics on Dataset {dataset}\n")
                f.write(f"- Precision: {metrics['precision']:.8f}\n")
                f.write(f"- Recall: {metrics['recall']:.8f}\n")
                f.write(f"- F1 Score: {metrics['f1']:.8f}\n")
                f.write(f"- True Positives: {metrics['tp']}\n")
                f.write(f"- False Positives: {metrics['fp']}\n")
                f.write(f"- False Negatives: {metrics['fn']}\n\n")
            elif task == "RelExt":
                f.write(f"\n### Overall Metrics on Dataset {dataset}\n")
                f.write(f"- Precision: {metrics['precision']:.8f}\n")
                f.write(f"- Recall: {metrics['recall']:.8f}\n")
                f.write(f"- F1 Score: {metrics['f1']:.8f}\n")
                f.write(f"- True Positives: {metrics['tp']}\n")
                f.write(f"- False Positives: {metrics['fp']}\n")
                f.write(f"- False Negatives: {metrics['fn']}\n\n")


def create_main_dashboard(results: Dict[str, Dict[str, Dict]], task: str, dataset: str):
    if task == 'EntExt':
        fig = make_subplots(rows=1, cols=1, subplot_titles=("Overall Metrics"))
        models = list(results.keys())
        precision = [results[model]['precision'] for model in models]
        recall = [results[model]['recall'] for model in models]
        f1 = [results[model]['f1'] for model in models]
        fig.add_trace(go.Bar(x=models, y=precision, name='Precision'), row=1, col=1)
        fig.add_trace(go.Bar(x=models, y=recall, name='Recall'), row=1, col=1)
        fig.add_trace(go.Bar(x=models, y=f1, name='F1 Score'), row=1, col=1)
        fig.update_layout(height=500, width=800, title_text=f"Model Performance Dashboard for Task EntExt on Dataset {dataset}")
    elif task == "RelExt":
        fig = make_subplots(rows=1, cols=1, subplot_titles=("Overall Metrics"))
        models = list(results.keys())
        precision = [results[model]['precision'] for model in models]
        recall = [results[model]['recall'] for model in models]
        f1 = [results[model]['f1'] for model in models]
        fig.add_trace(go.Bar(x=models, y=precision, name='Precision'), row=1, col=1)
        fig.add_trace(go.Bar(x=models, y=recall, name='Recall'), row=1, col=1)
        fig.add_trace(go.Bar(x=models, y=f1, name='F1 Score'), row=1, col=1)
        fig.update_layout(height=500, width=800, title_text=f"Model Performance Dashboard for Task RelExt on Dataset {dataset}")
    return fig


def load_results(task: str, dataset: str):
    data = defaultdict(lambda: list())
    for file in glob.glob(f"./results/{task}/*.json"):
        if task in file and dataset in file:
            with open(file, "r") as f:
                model_name = file.split("/")[-1].split("__")[2]
                data[model_name] = json.load(f)
    return data

def main(task: str, dataset: str):
    if task == "EntExt":
        if dataset == "entrelbench":
            dataset = "entrelbench_entities"
        elif dataset == "userstudy":
            dataset = "user_study_entities"
    elif task == "RelExt":
        if dataset == "entrelbench":
            dataset = "entrelbench_relations"
        elif dataset == "userstudy":
            dataset = "user_study_relations"
    data = load_results(task=task, dataset=dataset)
    if task == 'EntExt':
        results = {model: calculate_metrics_EntExt(predictions) for model, predictions in data.items()}
    elif task == 'RelExt':
        results = {model: calculate_metrics_RelExt(predictions) for model, predictions in data.items()}

    # Write results to markdown file
    write_markdown(results, f'Results_{task}.md', task, dataset)

    # Create and display the main dashboard
    main_dashboard = create_main_dashboard(results, task, dataset)
    main_dashboard.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process results for different tasks.')
    parser.add_argument('task', type=str, choices=['EntExt', 'RelExt'], help='Task to process')
    parser.add_argument('dataset', type=str, choices=['entrelbench', 'userstudy'], help='Task to process')

    args = parser.parse_args()


    main(args.task, args.dataset)