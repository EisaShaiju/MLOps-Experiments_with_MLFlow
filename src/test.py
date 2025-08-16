import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
print(mlflow.get_tracking_uri())