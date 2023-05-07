import json
from pathlib import Path

import pandas as pd


def merge_multiple_dataframe(input_folder_path: Path, output_folder_path: str):
    # check for datasets, compile them together, and write to an output Path
    print(f"Find and concatenate csvs in {input_folder_path}")
    df = pd.DataFrame()
    ingest_files = []
    for csv_file in input_folder_path.glob("*.csv"):
        ingest_files.append(str(csv_file))
        df = pd.concat([df, pd.read_csv(csv_file)])

    print("Deduplicate results")
    df = df.drop_duplicates()

    final_csv = output_folder_path / "finaldata.csv"
    print(f"Save to {final_csv}")
    output_folder_path.mkdir(parents=True, exist_ok=True)
    df.to_csv(final_csv)

    ingest_data_csv = output_folder_path / "ingestedfiles.txt"
    print(f"Save record of ingested data to {ingest_data_csv}")
    with open(ingest_data_csv, "w") as f:
        f.write("\n".join(ingest_files))


if __name__ == "__main__":
    with open("config.json", "r") as f:
        config = json.load(f)

    input_folder_path = Path(config["input_folder_path"])
    output_folder_path = Path(config["output_folder_path"])

    merge_multiple_dataframe(input_folder_path, output_folder_path)
