from mlflow.tracking import MlflowClient
import warnings
warnings.filterwarnings('ignore')

# Initialize the MLflow client
client = MlflowClient()

# Get the experiment ID from the MLflow experiment registry
experiment_id = client.get_experiment_by_name('dvc_retrain_model').experiment_id

# Get the latest version of the 'Airline Passenger Satisfaction Classifier' model
latest_version = client.get_latest_versions(name='Airline Passenger Satisfaction Classifier', stages=["None"])[0].version

# Load the model from the MLflow Model Registry (Production stage)
model = client.get_model_version(name='Airline Passenger Satisfaction Classifier', version=latest_version)

# Print the model details
model_name = model.name

client.transition_model_version_stage(name=model_name,
                                      version=latest_version,
                                      stage='Staging',
                                      archive_existing_versions=False)

print(f"Model '{model_name}' version: {latest_version} transitioned successfully from 'None' to 'Staging'.")