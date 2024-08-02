# entrelbench

A Benchmarking Pipeline for Cyber Security Entities and Relations Extraction

## Getting Started
- Install dependencies: `pip install -r requirements.txt`

## Run Benchmarking

### Entity Extraction:
```
python3 main.py --dataset_path entrelbench_entities.json --task EntExt --model mistralai/Mistral-7B-Instruct-v0.3 --max_tokens 2048 --temperature 0.7
```

### Relation Extraction:
```
python3 main.py --dataset_path entrelbench_relations.json --task RelExt --model mistralai/Mistral-7B-Instruct-v0.3 --max_tokens 2048 --temperature 0.7
```

### Few Notes:
- `settings.json` file contains the default connection and sampling parameters.
- `example.env` file contains secrets to use some commercial LLMs.
- `singature.py` file contains prompts format with DSPy signatures.
- `instructions.json` file contains instructions of our prompts.

