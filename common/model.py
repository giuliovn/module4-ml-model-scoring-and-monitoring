from sklearn.metrics import fbeta_score, precision_score, recall_score, confusion_matrix


def evaluate_regression_model(Y_data, y_pred):
    precision = precision_score(Y_data, y_pred, zero_division=1)
    recall = recall_score(Y_data, y_pred, zero_division=1)
    fbeta = fbeta_score(Y_data, y_pred, beta=1, zero_division=1)
    return precision, recall, fbeta


def make_confusion_matrix(Y_data, y_pred):
    return confusion_matrix(Y_data, y_pred)


def inference(model, prediction_data):
    return model.predict(prediction_data)
