import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, precision_recall_curve, roc_auc_score)
from sklearn.preprocessing import TargetEncoder, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as IbmPipeline
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import ADASYN, SMOTE

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

path = "C:/Users/Acer/Downloads/original_customer_data.csv"
df = pd.read_csv(path)

recently_joined = df[(df['Customer_Status'] == 'Joined')]  # .dropna(axis= 'columns', how= 'all')
stay_and_churn = df[(df['Customer_Status'] != 'Joined')]  # .dropna(axis= 'columns', how= 'all')

data = stay_and_churn.drop(['Customer_ID', 'Churn_Category', 'Churn_Reason'], axis=1)
numerical_columns = stay_and_churn.select_dtypes(include=['int64', 'float64']).columns.to_list()
ohe_columns = ['Gender', 'Married', 'Phone_Service', 'Multiple_Lines', 'Internet_Service', 'Internet_Type',
               'Online_Security', 'Online_Backup', 'Device_Protection_Plan', 'Premium_Support',
               'Streaming_TV', 'Streaming_Movies', 'Streaming_Music', 'Unlimited_Data', 'Contract',
               'Paperless_Billing', 'Payment_Method']
ordinal_columns = ['Value_Deal']
te_columns = ['State']

data['Customer_Status'] = data['Customer_Status'].map({'Stayed': 0, 'Churned': 1})

X = data.drop('Customer_Status', axis=1)
y = data['Customer_Status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_classifier_model = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=None)

preprocessor = ColumnTransformer(transformers=[
    ('cat', TargetEncoder(smooth=1.0), te_columns),
    ('onehot', OneHotEncoder(), ohe_columns),
    ('ord', OrdinalEncoder(categories=[['Deal 1', 'Deal 2', 'Deal 3', 'Deal 4', 'Deal 5', np.nan]]),
     ordinal_columns),
    ('num', 'passthrough', numerical_columns)
])


pipeline = Pipeline(steps=[
    ('encodings', preprocessor),
    #('adasyn', ADASYN(sampling_strategy= 'auto', random_state = 42)),
    #('smote', SMOTETomek(random_state = 42)),
    ('model', rf_classifier_model)])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

print("Confusion Matrix for pre-tuning vanilla RFC:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report for pre-tuning vanilla RFC:")
print(classification_report(y_test, y_pred))
print("AUC: ", roc_auc_score(y_test, y_pred_proba))

precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)

f1_scores = 2 * ((precision * recall) / (precision + recall))
f1_scores = np.nan_to_num(f1_scores, nan=0.0)

print('Best Threshold: ', thresholds[np.argmax(f1_scores)])
print('Best F1-Score: ', np.max(f1_scores))
