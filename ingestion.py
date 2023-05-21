import json
from pathlib import Path

from common.data import merge_multiple_dataframe


def ingest(input_folder_path, output_folder_path):
    merge_multiple_dataframe(input_folder_path, output_folder_path)


if __name__ == "__main__":
    with open("config.json", "r") as f:
        config = json.load(f)

    input_folder_path = Path(config["input_folder_path"])
    output_folder_path = Path(config["output_folder_path"])
    ingest(input_folder_path, output_folder_path)
