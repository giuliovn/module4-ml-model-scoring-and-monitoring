import json
from pathlib import Path

import pandas as pd

from common.common import save_file


def merge_multiple_dataframe(input_folder_path: Path, output_folder_path: str = None):
    # check for datasets, compile them together, and write to an output Path
    print(f"Find and concatenate csvs in {input_folder_path}")
    df = pd.DataFrame()
    ingest_files = []
    for csv_file in input_folder_path.glob("*.csv"):
        ingest_files.append(str(csv_file))
        df = pd.concat([df, pd.read_csv(csv_file)], ignore_index=True)

    print("Deduplicate results")
    df = df.drop_duplicates()

    if output_folder_path:
        final_csv = output_folder_path / "finaldata.csv"
        print(f"Save to {final_csv}")
        save_file(df, final_csv, format="csv")

        ingest_data_txt = output_folder_path / "ingestedfiles.txt"
        save_file(ingest_files, ingest_data_txt, format="txt")

    return df


if __name__ == "__main__":
    with open("config.json", "r") as f:
        config = json.load(f)

    input_folder_path = Path(config["input_folder_path"])
    output_folder_path = Path(config["output_folder_path"])

    merge_multiple_dataframe(input_folder_path, output_folder_path)
