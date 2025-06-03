import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import joblib, logging, pytest, warnings, json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
warnings.filterwarnings('ignore')

pipeline = joblib.load('models/model.pkl')

# Configure the logger
logging.basicConfig(filename='outputs/model_testing.log',level=logging.INFO,format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('model_testing')

df = pd.read_csv('data/test.csv')
df.drop('Unnamed: 0',axis=1,inplace=True)

def one_hot_encode(data,col):
    encoder = OneHotEncoder(drop='first',sparse_output=False,max_categories=5,dtype=int)
    encoded_df = encoder.fit_transform(data[[col]])
    encoded_df = pd.DataFrame(encoded_df,columns=encoder.get_feature_names_out())
    data = pd.concat([data,encoded_df],axis=1)
    data.drop(col,axis=1,inplace=True)
    return data

outlier_cols = ['Flight Distance','Arrival Delay in Minutes','Departure Delay in Minutes']
categorical_cols = [col for col in df.select_dtypes(object).columns if col != 'satisfaction']

for col in categorical_cols:
    df = one_hot_encode(df,col)

def impute_outliers(data,col):
    lower_limit, upper_limit = data[col].quantile([0.25,0.75])
    IQR = upper_limit - lower_limit
    lower_whisker = lower_limit - 1.5 * IQR
    upper_whisker = upper_limit + 1.5 * IQR
    return np.where(data[col]<lower_whisker,lower_whisker,np.where(data[col]>upper_whisker,upper_whisker,data[col]))

for col in outlier_cols:
    df[col] = impute_outliers(df,col)

df['Average service rating'] = df[['On-board service','Leg room service','Checkin service','Inflight service','Cleanliness','Departure/Arrival time convenient','Food and drink','Inflight entertainment','Seat comfort','Baggage handling','Inflight wifi service']].mean(axis=1)

final_selected_features = ['Inflight wifi service',
                           'Flight Distance',
                           'Type of Travel_Personal Travel',
                           'Average service rating',
                           'Online boarding',
                           'Class_Eco',
                           'Inflight entertainment',
                           'Seat comfort',
                           'Age',
                           'Customer Type_disloyal Customer']
if 'satisfaction' not in df.columns:
    raise ValueError("The target column 'satisfaction' is not present in the test data. We need ground truth data to analyze model performance on the test dataset.")

encoder = LabelEncoder()
df.satisfaction = encoder.fit_transform(df.satisfaction)

X_test = df.drop('satisfaction',axis=1)
y_test = df.satisfaction

X_test = X_test[final_selected_features]
y_pred = pipeline.predict(X_test)

acc = accuracy_score(y_pred,y_test)
precision = precision_score(y_pred,y_test)
recall = recall_score(y_pred,y_test)
f1 = f1_score(y_pred,y_test)
roc_auc = roc_auc_score(y_pred,y_test)

metrics = {
    'accuracy': acc,
    'precision': precision,
    'recall': recall,
    'f1': f1,
    'roc-auc': roc_auc
}

with open('outputs/metrics.json','w') as f:
    json.dump(metrics,f)

test_cases = [
    {"input": X_test[i:i+1], "expected_output": y_test.iloc[i]}
    for i in range(50)
]

@pytest.mark.parametrize("test_input, expected_output",
                         [(tc['input'],tc['expected_output']) for tc in test_cases])
def test_model_predictions(test_input,expected_output):
    # Make predictions
    prediction = pipeline.predict(test_input)[0]

    # Log the prediction and expected output
    logger.info(
        "Prediction: %s, Expected: %s, Match: %s",
        prediction,
        expected_output,
        prediction == expected_output
    )

