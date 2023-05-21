import json
from pathlib import Path
from shutil import copy2


def deploy(files_to_deploy: list[Path], deploy_dir: Path):
    deploy_dir.mkdir(parents=True, exist_ok=True)
    for file in files_to_deploy:
        print(f"Copy {file} to {deploy_dir}")
        copy2(file, deploy_dir)


if __name__ == "__main__":
    with open("config.json", "r") as f:
        config = json.load(f)

    output_folder_path = Path(config["output_folder_path"])
    output_model_path = Path(config["output_model_path"])
    prod_deployment_dir = Path(config["prod_deployment_path"])
    files_to_deploy = [
        output_model_path / "latestscore.txt",
        output_folder_path / "ingestedfiles.txt",
        output_model_path / "encoder.pkl",
        output_model_path / "trainedmodel.pkl",
    ]
    deploy(files_to_deploy, prod_deployment_dir)
