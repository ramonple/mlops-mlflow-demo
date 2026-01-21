import mlflow
from mlflow.models import infer_signature

import pandas as pd 
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

mlflow.set_tracking_uri("http://localhost:5000")

X, y = datasets.load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

params = {
    "solver": "liblinear",
    "max_iter": 200,
    "multi_class":"auto",
    "random_state":42
}

lr = LogisticRegression(**params)
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

accuray = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

# set our tracking server uri for logging
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# create a new mlflow experiment
mlflow.set_experiment("ML flow Quickstart")

with mlflow.start_run():
    # log hyperparameters
    mlflow.log_params(params)
    
    # log metrics
    mlflow.log_metric("accuracy", accuray)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    
    # set a tag that we can use to remind ourselves what this run was about
    mlflow.set_tag("Training Info", "Basic Logistic Regression model on Iris dataset")

    # infer model signature
    signature = infer_signature(X_train, lr.predict(X_train))
    
    # log the model
    model_info = mlflow.sklearn.log_model(
        lr,
        artifact_path="logistic-regression-model",
        signature=signature
    )

# load the model for inference
loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)

# make predictions
predictions = loaded_model.predict(X_test)
print("Predictions:", predictions)  

iris_features_names = datasets.load_iris().feature_names

results = pd.DataFrame(X_test, columns=iris_features_names)
result['Actual Class'] = y_test
results["Predicted Class"] = predictions

results[:4]


