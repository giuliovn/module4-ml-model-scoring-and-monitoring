import json
from pathlib import Path
from shutil import copyfile


def store_model_into_pickle(results_dir: Path, deploy_dir: Path):
    files_to_deploy = [
        "latestscore.txt",
        "ingestedfiles.txt",
        "encoder.pkl",
        "trainedmodel.pkl",
    ]
    for file in files_to_deploy:
        print(f"Copy {file} to {deploy_dir}")
        copyfile(results_dir / file, deploy_dir / file)


if __name__ == "__main__":
    with open("config.json", "r") as f:
        config = json.load(f)

    results_dir = Path(config["output_folder_path"])
    prod_deployment_dir = Path(config["prod_deployment_path"])
    prod_deployment_dir.mkdir(parents=True, exist_ok=True)

    store_model_into_pickle(results_dir, prod_deployment_dir)
