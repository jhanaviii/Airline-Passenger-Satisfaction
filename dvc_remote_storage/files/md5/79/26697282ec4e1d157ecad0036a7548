from flask import Flask, render_template, request
import pandas as pd
import joblib, warnings
warnings.filterwarnings('ignore')
from mlflow.pyfunc import load_model
from mlflow.tracking import MlflowClient

# Initialize the MLflow client
client = MlflowClient(tracking_uri="http://127.0.0.1:5000")

# Name of the registered model
registered_model_name = "Airline Passenger Satisfaction Classifier"
run_id = "" # Placeholder for the run_id

# Fetch the latest version of the model in the 'Production' stage
latest_versions = client.get_latest_versions(name=registered_model_name, stages=['Production'])

if latest_versions:
    latest_version = latest_versions[0].version # Get the latest version
    run_id = latest_versions[0].run_id # Fetch the run_id of the latest version
else:
    print(f"No versions of the model '{registered_model_name}' found in the specified stage.")
    run_id = "4c65882de40f46549fa2a40202513093" # Otherwise setting a default run_id

try:
    # registered_model_name = 'Airline Passenger Satisfaction Classifier'
    # client = MlflowClient()
    # registered_model = client.get_registered_model(name=registered_model_name)
    # latest_version = registered_model.latest_versions[0].version
    # # Load the model from the MLflow Model Registry (Production stage)
    # model = load_model(
    #     model_uri=f"models:/{registered_model_name}/{latest_version}"
    # )
    model_uri = f"runs:/{run_id}/{registered_model_name}"
    print(f"Successfully loaded the model '{registered_model_name}' from 'Production' stage.")
except Exception as e:
    print(f"Error loading the model: {e}")
    raise

pipeline = joblib.load('model.pkl')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        form_inputs = ['inflight_wifi_service',
                           'flight_distance',
                           'type_of_travel_personal',
                           'average_service_rating',
                           'online_boarding',
                           'class_economy',
                           'inflight_entertainment',
                           'seat_comfort',
                           'age',
                           'customer_type_disloyal']
        features = [float(request.form.get(x)) for x in form_inputs]
        feature_names = ['Inflight wifi service',
                           'Flight Distance',
                           'Type of Travel_Personal Travel',
                           'Average service rating',
                           'Online boarding',
                           'Class_Eco',
                           'Inflight entertainment',
                           'Seat comfort',
                           'Age',
                           'Customer Type_disloyal Customer']
        data = pd.DataFrame([features],columns=feature_names)
        prediction = pipeline.predict(data)[0]
        if prediction == 0:
            return render_template('index.html', prediction_text="The passenger is likely not satisfied with the airline's services.")
        else:
            return render_template('index.html', prediction_text="The passenger is satisfied with services provided by the airline.")

if __name__ == '__main__':
    app.run(port=2000)