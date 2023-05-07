import joblib
import numpy as np
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.preprocessing import OneHotEncoder


def process_data(X, categorical_features=[], label=None, training=True, encoder=None):
    if label is not None:
        y = X[label]
        X = X.drop([label], axis=1)
    else:
        y = np.array([])

    X_categorical = X[categorical_features].values
    X_continuous = X.drop(*[categorical_features], axis=1)

    if training is True:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        X_categorical = encoder.fit_transform(X_categorical)
    else:
        X_categorical = encoder.transform(X_categorical)

    X = np.concatenate([X_continuous, X_categorical], axis=1)
    return X, y, encoder


def evaluate_regression_model(model, X_data, Y_data):
    y_pred = model.predict(X_data)
    precision = precision_score(Y_data, y_pred, zero_division=1)
    recall = recall_score(Y_data, y_pred, zero_division=1)
    fbeta = fbeta_score(Y_data, y_pred, beta=1, zero_division=1)
    return precision, recall, fbeta


def save_file(data, output_path, format="pkl"):
    # format can be pkl, csv or txt
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
