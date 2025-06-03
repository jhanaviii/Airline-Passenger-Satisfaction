from mlflow.tracking import MlflowClient
import warnings, mlflow
warnings.filterwarnings('ignore')

run_id = "4c65882de40f46549fa2a40202513093"

model_path = "file:///Users/sayamkumar/Desktop/Data%20Science%20Projects/Airline%20Passenger%20Satisfaction/mlruns/594113527903238029/4c65882de40f46549fa2a40202513093/artifacts/model"
model_name = "Airline Passenger Satisfaction Classifier"

model_uri = f"runs:/{run_id}/{model_path}"

# Register the model in the MLflow Model Registry
registered_model = mlflow.register_model(model_uri=model_uri,name=model_name)
print(f"Model '{model_name}' with version: {registered_model.version} registered successfully in the Model Registry.")

