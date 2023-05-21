from glob import glob
import json
from pathlib import Path

from ingestion import ingest
from scoring import score_model
from training import train_model
from deployment import deploy
from reporting import model_report
from apicalls import call_apis

with open("config.json", "r") as f:
    config = json.load(f)

input_folder_path = Path(config["input_folder_path"])
output_folder_path = Path(config["output_folder_path"])
deployment_dir = Path(config["prod_deployment_path"])
ingested_data_file = deployment_dir / "ingestedfiles.txt"
print(
    f"Read {ingested_data_file} and check if new data is available in {input_folder_path}"
)
with open(ingested_data_file, "r") as f:
    ingested_data_list = f.read().strip().splitlines()

new_data = glob(str(input_folder_path / "*.csv"))
if sorted(ingested_data_list) != sorted(new_data):
    print("New data found. Check if retrain is needed")
else:
    print("No new data")
    exit(0)

ingest(input_folder_path, output_folder_path)

latest_score_file = deployment_dir / "latestscore.txt"
print(f"Read previous model score from {latest_score_file}")
with open(latest_score_file, "r") as f:
    latest_score = f.read().strip()

deployed_model_path = deployment_dir / "trainedmodel.pkl"
deployed_encoder_path = deployment_dir / "encoder.pkl"
categorical_features = config["categorical_features"]
Y_label = config["Y_label"]
new_data_file = output_folder_path / "finaldata.csv"
output_model_path = Path(config["output_model_path"])
latest_score_path = output_model_path / "latestscore.txt"
print("Score model on new data")
fbeta = score_model(
    deployed_model_path,
    deployed_encoder_path,
    new_data_file,
    categorical_features,
    Y_label,
    latest_score_path,
)
print(f"New model: {fbeta}. Latest model score: {latest_score}")
if float(fbeta) < float(latest_score):
    print("Model perform bad on new data. Retrain")
else:
    print("Model perform well on new data")
    exit(0)

train_model(new_data_file, categorical_features, Y_label, output_model_path)

files_to_deploy = [
    output_model_path / "latestscore.txt",
    output_folder_path / "ingestedfiles.txt",
    output_model_path / "encoder.pkl",
    output_model_path / "trainedmodel.pkl",
]
deploy(files_to_deploy, deployment_dir)

print("Run model report")
model_report(
    deployment_dir / "trainedmodel.pkl",
    deployment_dir / "encoder.pkl",
    new_data_file,
    categorical_features,
    Y_label,
    output_model_path,
)

print("Call live apis")
call_apis(output_model_path)
