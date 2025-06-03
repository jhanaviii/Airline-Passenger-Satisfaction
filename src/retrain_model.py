import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import BaggingClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier, VotingClassifier, RandomForestClassifier, HistGradientBoostingClassifier, StackingClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier, SGDClassifier, RidgeClassifier
from xgboost import XGBClassifier, XGBRFClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from ngboost import NGBClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neural_network import MLPClassifier
from lightgbm import LGBMClassifier  
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
import joblib, warnings, mlflow
warnings.filterwarnings('ignore')

# Load the train dataset
df = pd.read_csv('data/train.csv')
df.drop('Unnamed: 0',axis=1,inplace=True)

def one_hot_encode(data,col):
    """
        Custom function to one-hot encode categorical variables.
        Args:
            data (pandas DataFrame): DataFrame containing the column to be processed.
            col (str): Column name to process.
    """
    encoder = OneHotEncoder(drop='first',sparse_output=False,max_categories=5,dtype=int)
    encoded_df = encoder.fit_transform(data[[col]])
    encoded_df = pd.DataFrame(encoded_df,columns=encoder.get_feature_names_out())
    data = pd.concat([data,encoded_df],axis=1)
    data.drop(col,axis=1,inplace=True)
    return data

outlier_cols = ['Flight Distance','Arrival Delay in Minutes','Departure Delay in Minutes']
categorical_cols = [col for col in df.select_dtypes(object).columns if col != 'satisfaction']

# One-hot encoding categorical variables
for col in categorical_cols:
    df = one_hot_encode(df,col)

def impute_outliers(data,col):
    """
        Customer function to handle outliers using the Interquartile Range (IQR) method.
        Args:
            data (pandas DataFrame): DataFrame containing the column to be processed.
            col (str): Column name to process.
    """
    lower_limit, upper_limit = data[col].quantile([0.25,0.75])
    IQR = upper_limit - lower_limit
    lower_whisker = lower_limit - 1.5 * IQR
    upper_whisker = upper_limit + 1.5 * IQR
    return np.where(data[col]<lower_whisker,lower_whisker,np.where(data[col]>upper_whisker,upper_whisker,data[col]))

for col in outlier_cols:
    df[col] = impute_outliers(df,col)

df['Average service rating'] = df[['On-board service','Leg room service','Checkin service','Inflight service','Cleanliness','Departure/Arrival time convenient','Food and drink','Inflight entertainment','Seat comfort','Baggage handling','Inflight wifi service']].mean(axis=1)

# Encoding the target variable
encoder = LabelEncoder()
df.satisfaction = encoder.fit_transform(df.satisfaction)

# Dividing the dataset into features (X) and target variable (y)
X = df.drop('satisfaction',axis=1)
y = df.satisfaction

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=101)

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

final_X_train = X_train[final_selected_features]
final_X_test = X_test[final_selected_features]

# Feature scaling
scaler = StandardScaler()
features = final_X_train.columns
final_X_train = scaler.fit_transform(final_X_train)
final_X_train = pd.DataFrame(final_X_train,columns=features)
final_X_test = scaler.transform(final_X_test)
final_X_test = pd.DataFrame(final_X_test,columns=features)

trained_models = []
accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []
roc_auc_scores = []

# Get MLFLow experiment details by name
# experiment_id = mlflow.create_experiment(name='dvc_retrain_model')
experiment = mlflow.get_experiment_by_name('dvc_retrain_model')

def train_and_evaluate_model(model):
    """
        Custom function to train and evaluate a model.
        Args:
            model (object): A scikit-learn model object.
    """
    model.fit(final_X_train,y_train)
    y_pred = model.predict(final_X_test)
    acc = accuracy_score(y_test,y_pred)
    prec = precision_score(y_test,y_pred)
    rec = recall_score(y_test,y_pred)
    f1 = f1_score(y_test,y_pred)
    roc = roc_auc_score(y_test,y_pred)
    trained_models.append(model)
    accuracy_scores.append(acc)
    precision_scores.append(prec)
    recall_scores.append(rec)
    f1_scores.append(f1)
    roc_auc_scores.append(roc)
    cm = confusion_matrix(y_pred,y_test)

    labels = ['satisfied','neutral or dissatisfied']
    # Log metrics and artifacts using MLflow
    with mlflow.start_run(experiment_id=experiment.experiment_id,run_name="Model Training"):
        # Log confusion matrix as an artifact
        plt.figure(figsize=(6, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')

        # Save the plot to a file
        cm_file_path = "confusion_matrix.png"
        plt.savefig(cm_file_path)
        plt.close()

        mlflow.log_artifact(cm_file_path)

        mlflow.log_metrics({
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "roc_auc_score": roc
        })

        mlflow.log_param("model_name", model.__class__.__name__)
    
train_and_evaluate_model(LogisticRegression())
train_and_evaluate_model(PassiveAggressiveClassifier())
train_and_evaluate_model(SGDClassifier())
train_and_evaluate_model(RidgeClassifier())
train_and_evaluate_model(KNeighborsClassifier())
train_and_evaluate_model(SVC())
train_and_evaluate_model(LinearSVC())
train_and_evaluate_model(DecisionTreeClassifier())
train_and_evaluate_model(ExtraTreesClassifier())
train_and_evaluate_model(RandomForestClassifier())
train_and_evaluate_model(BaggingClassifier())
train_and_evaluate_model(GaussianNB())
train_and_evaluate_model(BernoulliNB())
train_and_evaluate_model(MLPClassifier())
train_and_evaluate_model(HistGradientBoostingClassifier())
train_and_evaluate_model(GradientBoostingClassifier())
train_and_evaluate_model(AdaBoostClassifier())
train_and_evaluate_model(NGBClassifier())
train_and_evaluate_model(CatBoostClassifier(silent=True))
train_and_evaluate_model(LGBMClassifier())
train_and_evaluate_model(XGBClassifier())
train_and_evaluate_model(XGBRFClassifier())
train_and_evaluate_model(VotingClassifier(estimators=[
    ('CAT',CatBoostClassifier(silent=True)),
    ('BAG',BaggingClassifier()),
    ('LGBM',LGBMClassifier()),
    ('HGB',HistGradientBoostingClassifier()),
    ('XGB',XGBClassifier())
],voting='hard'))
train_and_evaluate_model(StackingClassifier(estimators=[
    ('CAT',CatBoostClassifier(silent=True)),
    ('BAG',BaggingClassifier()),
    ('RF',RandomForestClassifier()),
    ('HGB',HistGradientBoostingClassifier()),
    ('XGB',XGBClassifier())
],final_estimator=LGBMClassifier(),cv=5,verbose=1))

# Determine the best model based on accuracy
model_performances = pd.DataFrame({'Model': trained_models, \
                                  'Accuracy': accuracy_scores, \
                                  'Precision': precision_scores, \
                                  'Recall': recall_scores, \
                                  'F1': f1_scores, \
                                  'ROC-AUC': roc_auc_scores}).sort_values('Accuracy',ascending=False).reset_index(drop=True)

best_model = model_performances.iloc[0]['Model']
y_pred = best_model.predict(final_X_test)
acc = accuracy_score(y_pred,y_test)
prec = precision_score(y_pred,y_test)
rec = recall_score(y_pred,y_test)
f1 = f1_score(y_pred,y_test)
roc = roc_auc_score(y_test,y_pred)
cv_acc = cross_val_score(estimator=best_model,X=final_X_test,y=y_test,cv=5,scoring='accuracy').mean()

# Load thresholds from the JSON file
# with open('outputs/accuracy_thresholds.json', 'r') as f:
#     stored_thresholds = json.load(f)

# baseline_acc_threshold = stored_thresholds['baseline_acc_threshold']
# baseline_cv_acc_threshold = stored_thresholds['baseline_cv_acc_threshold']

print("Confusion Matrix:")
print(confusion_matrix(y_test,y_pred))
print("Classification Report:")
print(classification_report(y_test,y_pred))

cm = confusion_matrix(y_pred,y_test)
labels = ['satisfied','neutral or dissatisfied']

with mlflow.start_run(experiment_id=experiment.experiment_id,run_name='best_model_evaluation') as run:
    # Log validation metrics 
    mlflow.log_metrics({
        "accuracy": acc,
        "cross_validation_accuracy": cv_acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc_score": roc
    })
    # Log confusion matrix as an artifact
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')

    # Save the plot to a file
    cm_file_path = "confusion_matrix.png"
    plt.savefig(cm_file_path)
    plt.close()

    mlflow.log_artifact(cm_file_path)

    # Log classification report as an artifact
    with open('classification_report.txt', 'w') as f:
        f.write(classification_report(y_test, y_pred))
    mlflow.log_artifact('classification_report.txt')
    # Log the best model - Stacking Classifier as an artifact
    artifact_path = "model"
    mlflow.sklearn.log_model(sk_model=best_model,
                             artifact_path=artifact_path,
                             registered_model_name='Airline Passenger Satisfaction Classifier')
    
# Create a pipeline for deployment and save it as a pickle file
pipeline = Pipeline(steps=[
                        ('scaler',scaler),
                        ('model',best_model)
                    ])

joblib.dump(pipeline, 'models/model.pkl')

# if acc > baseline_acc_threshold and cv_acc > baseline_cv_acc_threshold:
#     print("New model exceeds thresholds. Updating model and thresholds.")    
#     # joblib.dump(pipeline,'models/airline_passenger_satisfaction_classifier.pkl')
#     joblib.dump(pipeline, temp_model_path)
#     # Update and save new thresholds
#     new_thresholds = {
#         'baseline_acc_threshold': acc,
#         'baseline_cv_acc_threshold': cv_acc
#     }
#     with open('outputs/accuracy_thresholds.json', 'w') as f:
#         json.dump(new_thresholds, f)

# else:
#     print("Current model did not exceed thresholds. No update made.")
#     if not os.path.exists('models/model.pkl'):
#         print("Old model does not exist. Saving current model as fallback.")
#         os.replace(temp_model_path, 'models/model.pkl')