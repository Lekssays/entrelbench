import dspy
import json
import signatures
import logging
import time
import os
import re

from datetime import datetime
from typing import List, Dict, Any
from adapters.DeepSeek import DeepSeek

SETTINGS_PATH = "./settings.json"
DATASETS_PATH = "./datasets/"


def parse_response(response: str, task: str) -> Any:
    """Parse the datapoint format"""
    
    parsers = {
        "EntExt": lambda r: list(set(r.replace("<entities>", "").replace("</entities>", "").split("|"))),
        "RelExt": lambda r: list(set(r.replace("<relations>", "").replace("</relations>", "").split("|"))),
    }
    
    return parsers.get(task, lambda r: r)(response)


def load_dataset(dataset_path: str, experiment_name: str) -> List[Dict[str, Any]]:
    """Load and preprocess the dataset for the experiment."""
    with open(DATASETS_PATH + dataset_path, 'r') as f:
        dataset = json.load(f)
    
    return [
        {
            "instruction": entry['INSTRUCTION'],
            "input": entry['INPUT'],
            "gold": parse_response(entry['OUTPUT'], task=experiment_name),
            "predicted": "",
        }
        for entry in dataset
    ]

def get_model_info(model_name: str) -> Dict[str, Any]:
    """Retrieve model information from settings."""
    with open(SETTINGS_PATH, 'r') as f:
        settings = json.load(f)
    return next((model for model in settings["models"] if model['name'] == model_name), None)

def get_parameters(experiment_name: str, client: str) -> Dict[str, Any]:
    """Get experiment parameters based on the client type."""
    with open(SETTINGS_PATH, 'r') as f:
        settings = json.load(f)
    
    params = settings["parameters"][experiment_name]
    if client in ["AzureOpenAI", "Claude"]:
        return {k: params[k] for k in ['max_tokens', 'top_p', 'temperature']}
    return params

def get_models() -> List[Dict[str, Any]]:
    """Retrieve all model configurations."""
    with open(SETTINGS_PATH, 'r') as f:
        return json.load(f)["models"]

def initialize_server(client: str, model: str, model_info: Dict[str, Any], config: dict) -> Any:
    """Initialize the appropriate server based on the client type."""
    if client == 'vLLM':
        return dspy.HFClientVLLM(model=model, url=model_info['host'], port=model_info['port'])
    elif client == 'Claude':
        return dspy.Claude(model=model, api_key=os.getenv("ANTHROPIC_API_KEY"))
    elif client == 'OpenAI':
        return dspy.OpenAI(model=model, api_key=os.getenv("OPENAI_API_KEY"), model_type="chat")
    elif client == 'DeepSeek':
        return DeepSeek(
            model=model,
            api_base=model_info["host"],
            api_key=os.getenv("DEEPSEEK_PALTFORM_API_KEY"),
            max_tokens=config["max_tokens"],
            temperature=config["temperature"],
            top_p=config["top_p"],
        )
    elif client == "AzureOpenAI":
        return dspy.AzureOpenAI(
            model=model,
            api_base=os.getenv("AZURE_OPENAI_API_ENDPOINT"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            api_key=os.getenv("AZURE_OPENAI_KEY"),
            model_type="chat"
        )
    elif client == "Ollama":
        return dspy.OllamaLocal(model=model, url=model_info['host'], port=model_info['port'])
    else:
        raise ValueError(f"Unsupported client: {client}")

def run_experiment(experiment_name: str, dataset_path: str, model: str, client: str, config=None):
    """Run the experiment with the specified parameters."""
    logging.info(f"Running experiment {experiment_name} with model {model} on dataset {dataset_path}")
    
    dataset = load_dataset(dataset_path=dataset_path, experiment_name=experiment_name)
    model_info = get_model_info(model_name=model)
    if not model_info:
        raise ValueError(f"Model '{model}' not found in settings file")
    
    if config is None:
        config = get_parameters(experiment_name=experiment_name, client=client)

    server = initialize_server(client=client, model=model, model_info=model_info, config=config)
    dspy.settings.configure(lm=server)

    predictor = dspy.Predict(getattr(signatures, experiment_name))
    
    results = []
    for i, datapoint in enumerate(dataset, 1):
        logging.info(f"Processing datapoint {i}/{len(dataset)}")
        try:
            pred = predictor(
                instruction=datapoint["instruction"],
                input=datapoint['input'],
                config=config
            )
            print("PRED.RESPONSE --->", )
            response = parse_response(response=pred.response, task=experiment_name)
            logging.info(f"Predicted: {response} | Gold: {datapoint['gold']}")
            datapoint['predicted'] = response
            results.append(datapoint)
        except Exception as e:
            logging.exception(f"Error processing datapoint {i}: {e}")
        
        if client in ["AzureOpenAI", "Claude", "DeepSeek"] and i % 40 == 0:
            time.sleep(10)
    
    save_results(experiment_name, dataset_path, model, config, results)

def save_results(experiment_name: str, dataset_path: str, model: str, config: Dict[str, Any], results: List[Dict[str, Any]]):
    """Save the experiment results to a file."""
    dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
    top_k = config.get("top_k", "None")
    filename = f"raw__{experiment_name}__{model.replace('/', '_')}__{dataset_name}__{config['temperature']}_{config['top_p']}_{top_k}_{config['max_tokens']}__{datetime.now().timestamp()}.json"
    
    os.makedirs(f"./results/{experiment_name}", exist_ok=True)
    with open(f"./results/{experiment_name}/{filename}", 'w') as f:
        json.dump(results, f, indent=4)
    
    logging.info(f'Results saved to {filename}')
