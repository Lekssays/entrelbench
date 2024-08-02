import utils
import logging
import argparse

logging.root.handlers = []
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
    handlers=[logging.FileHandler("./entrelbench.log"), logging.StreamHandler()],
)


def main():
    parser = argparse.ArgumentParser(
        description="Run an experiment with specified parameters."
    )
    parser.add_argument(
        "--top_p",
        type=float,
        required=False,
        help="Top P value for the experiment",
        default=0.95,
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        required=False,
        help="Max tokens value for the experiment",
        default=50,
    )
    parser.add_argument(
        "--temperature",
        type=float,
        required=False,
        help="Temperature value for the experiment",
        default=0.0,
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["EntExt", "RelExt"],
        help="Task. See settings.json for all the possible names.",
    )
    parser.add_argument(
        "--dataset_path", type=str, required=True, help="Path to the dataset"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model to use for the experiment. See settings.json for all the possible models.",
    )

    args = parser.parse_args()

    top_p = args.top_p
    temperature = args.temperature
    max_tokens = args.max_tokens
    task = args.task
    dataset_path = args.dataset_path
    model = args.model
    client = utils.get_model_info(model_name=model)['client']

    config = None
    if top_p != 0.95 or temperature != 0.7 or max_tokens != 2048:
        config = {"top_p": top_p, "temperature": temperature, "max_tokens": max_tokens}

    logging.info(f"Starting experiment: {task} with model: {model}")
    logging.info(
        f"Parameters - Top P: {top_p}, Temperature: {temperature}, Dataset Path: {dataset_path}"
    )
    utils.run_experiment(
        experiment_name=task,
        model=model,
        dataset_path=dataset_path,
        config=config,
        client=client
    )


if __name__ == "__main__":
    main()