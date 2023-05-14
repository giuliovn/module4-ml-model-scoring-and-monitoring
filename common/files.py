import joblib


def save_file(data, output_path, format=None):
    # format can be pkl, csv or txt
    if not format:
        format = output_path.suffix.split(".")[-1]  # split to remove dot
    print(f"Save to {output_path} in {format} format")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if format == "csv":
        data.to_csv(output_path, index=False)
    if format == "pkl":
        joblib.dump(data, output_path)
    if format == "txt":
        with open(output_path, "w") as f:
            lines = "\n".join(data) if isinstance(data, list) else data
            f.write(lines)
