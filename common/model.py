from sklearn.metrics import fbeta_score, precision_score, recall_score


def evaluate_regression_model(model, X_data, Y_data):
    y_pred = inference(model, X_data)
    precision = precision_score(Y_data, y_pred, zero_division=1)
    recall = recall_score(Y_data, y_pred, zero_division=1)
    fbeta = fbeta_score(Y_data, y_pred, beta=1, zero_division=1)
    return precision, recall, fbeta


def inference(model, prediction_data):
    return model.predict(prediction_data)
