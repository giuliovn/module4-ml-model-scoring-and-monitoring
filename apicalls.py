import json
from pathlib import Path
import requests

from common.files import save_file

# Specify a URL that resolves to your workspace
URL = "http://127.0.0.1:8000/"


def call_apis(output_path):
    predict = requests.post(URL + "prediction?data=testdata/testdata.csv")
    score = requests.get(URL + "scoring")
    stats = requests.get(URL + "summarystats")
    diagnose = requests.get(URL + "diagnostics")

    # combine all API responses
    responses = json.dumps(
        {
            "prediction": predict.json(),
            "score": score.json(),
            "stats": stats.json(),
            "diagnose": diagnose.json(),
        },
        sort_keys=True,
        indent=4,
    )

    print(responses)
    save_file(responses, output_path / "apireturns.txt", format="txt")


if __name__ == "__main__":
    with open("config.json", "r") as f:
        config = json.load(f)

    output_path = Path(config["output_model_path"])

    call_apis(output_path)
